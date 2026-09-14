//! snowrun — headless Snow runner for the born-32 (hugeSE) ROM work.
//!
//! Boots a ROM with an optional SCSI hard disk and no floppy, the mouse
//! disabled and a pinned real-time clock, so that runs are repeatable. The only
//! input is scripted keypresses at fixed emulated times (--key). Writes a PNG of
//! the screen every N emulated seconds, plus a final one, logging the CPU's PC at
//! each. Optionally dumps guest RAM (--dump-ram), the last N instructions
//! (--history) and the last N A-line traps (--traps) at the end.
//!
//! Disk writes stay in memory: testrunner builds snow_core without the `mmap`
//! feature, so the image is read into a Vec and the file itself is never
//! modified.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use log::*;

use snow_core::cpu_m68k::cpu::{
    Breakpoint, HistoryEntry, HistoryEntryInstruction, SystrapHistoryEntry,
};
use snow_core::cpu_m68k::disassembler::Disassembler;
use snow_core::cpu_m68k::regs::RegisterFile;
use snow_core::emulator::comm::{EmulatorCommand, EmulatorEvent, EmulatorEventReceiver, EmulatorSpeed};
use snow_core::emulator::{Emulator, MouseMode};
use snow_core::keymap::{KeyEvent, Keymap};
use snow_core::mac::MacModel;
use snow_core::tickable::{Tickable, Ticks};
use snow_core::types::Long;

/// Compact-Mac bus clock: the compact bus advances one tick per 7.8336 MHz SE clock.
const SE_CLOCK_HZ: f64 = 7_833_600.0;

#[derive(Parser)]
#[command(about = "Headless Snow runner: ROM + SCSI disk, pinned clock, periodic screenshots")]
struct Args {
    /// ROM image
    rom: PathBuf,

    /// Output directory for screenshots and dumps
    out_dir: PathBuf,

    /// Mac model (skips ROM auto-detection)
    #[arg(long)]
    model: Option<MacModel>,

    /// SCSI hard disk image, attached at --scsi-id
    #[arg(long)]
    scsi: Option<PathBuf>,

    /// SCSI ID for --scsi
    #[arg(long, default_value_t = 0)]
    scsi_id: usize,

    /// RAM size in megabytes (model default if omitted)
    #[arg(long)]
    ram_mb: Option<usize>,

    /// Enable the PMMU (68020/030 models only)
    #[arg(long)]
    pmmu: bool,

    /// Emulated seconds to run
    #[arg(long, default_value_t = 60.0)]
    seconds: f64,

    /// Emulated seconds between screenshots (0 = final only)
    #[arg(long, default_value_t = 5.0)]
    shot_every: f64,

    /// Pinned real-time clock, "YYYY-MM-DD HH:MM:SS"
    #[arg(long, default_value = "1996-01-01 12:00:00")]
    date: String,

    /// Scripted keypress "SECS:KEY[:HOLD]", repeatable. KEY is a name (return,
    /// enter, shift, cmd, option, space, esc) or a hex Universal/ADB scancode
    /// such as 0x24. HOLD is how long it stays down, in seconds (default 0.1).
    #[arg(long = "key", value_name = "SECS:KEY[:HOLD]")]
    keys: Vec<String>,

    /// Write guest RAM at the end of the run to this file. Built from the core's
    /// dirty-page snapshots, so it is as current as the last status update.
    #[arg(long)]
    dump_ram: Option<PathBuf>,

    /// Record instruction history and write the last N instructions to
    /// OUT_DIR/history.txt at the end (0 = off; slows emulation)
    #[arg(long, default_value_t = 0)]
    history: usize,

    /// Record A-line trap history and write the last N traps to OUT_DIR/traps.txt
    /// at the end (0 = off)
    #[arg(long, default_value_t = 0)]
    traps: usize,

    /// Stop the emulator when the CPU takes this exception vector (vector address in
    /// hex, e.g. 0x10 = illegal instruction, 0x0C = address error), repeatable. The
    /// run then ends there and all dumps capture that moment.
    #[arg(long = "break-vector", value_name = "ADDR")]
    break_vectors: Vec<String>,
}

/// What the runner has seen from the emulator so far.
#[derive(Default)]
struct Observed {
    /// Guest RAM, mirrored from dirty-page snapshots
    ram: Vec<u8>,
    /// Register file from the latest status update
    regs: Option<RegisterFile>,
    /// Latest instruction history snapshot
    history: Vec<HistoryEntry>,
    /// Latest A-line trap history snapshot
    traps: Vec<SystrapHistoryEntry>,
}

fn write_png(path: &Path, width: u16, height: u16, rgba: &[u8]) -> Result<()> {
    let mut encoder = png::Encoder::new(File::create(path)?, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(rgba)?;
    Ok(())
}

/// Parses "SECS:KEY[:HOLD]" into (press time, Universal scancode, hold time).
fn parse_key(spec: &str) -> Result<(f64, u8, f64)> {
    let mut parts = spec.split(':');
    let secs: f64 = parts
        .next()
        .unwrap_or_default()
        .parse()
        .with_context(|| format!("--key '{spec}': bad SECS"))?;
    let key = parts
        .next()
        .with_context(|| format!("--key '{spec}': expected SECS:KEY"))?;
    let code = match key.to_ascii_lowercase().as_str() {
        "return" => 0x24,
        "enter" => 0x4C,
        "shift" => 0x38,
        "cmd" | "command" => 0x37,
        "option" => 0x3A,
        "space" => 0x31,
        "esc" | "escape" => 0x35,
        k => u8::from_str_radix(k.trim_start_matches("0x"), 16)
            .with_context(|| format!("--key '{spec}': unknown key '{key}'"))?,
    };
    let hold = match parts.next() {
        Some(h) => h
            .parse()
            .with_context(|| format!("--key '{spec}': bad HOLD"))?,
        None => 0.1,
    };
    Ok((secs, code, hold))
}

/// Drains pending emulator events into `obs`. Returns true if the emulator
/// reported that it stopped (e.g. halted on a double fault).
fn drain_events(event_recv: &EmulatorEventReceiver, obs: &mut Observed) -> bool {
    let mut stopped = false;
    while let Ok(event) = event_recv.try_recv() {
        match event {
            EmulatorEvent::Memory((addr, bytes, ram_len)) => {
                if obs.ram.len() != ram_len {
                    obs.ram.resize(ram_len, 0);
                }
                let start = addr as usize;
                obs.ram[start..start + bytes.len()].copy_from_slice(&bytes);
            }
            EmulatorEvent::Status(s) => {
                stopped |= !s.running && s.cycles > 100;
                obs.regs = Some(s.regs.clone());
            }
            EmulatorEvent::InstructionHistory(h) => obs.history = h,
            EmulatorEvent::SystrapHistory(t) => obs.traps = t,
            event => debug!("event: {}", event),
        }
    }
    stopped
}

/// One-line register summary for the log.
fn regs_summary(regs: Option<&RegisterFile>) -> String {
    regs.map_or_else(
        || "pc=?".to_string(),
        |r| {
            format!(
                "pc=${:08X} sr=${:04X} d0=${:08X} a0=${:08X} a6=${:08X} a7=${:08X}",
                r.pc,
                r.sr.0,
                r.read_d::<Long>(0),
                r.read_a::<Long>(0),
                r.read_a::<Long>(6),
                r.read_a::<Long>(7)
            )
        },
    )
}

/// Names the common 68k exception vectors (by vector address).
const fn vector_name(vector: u32) -> &'static str {
    match vector {
        0x08 => "bus error",
        0x0C => "address error",
        0x10 => "illegal instruction",
        0x14 => "zero divide",
        0x18 => "CHK",
        0x1C => "TRAPV",
        0x20 => "privilege violation",
        0x24 => "trace",
        0x28 => "line-A",
        0x2C => "line-F",
        0x38 => "format error",
        0x60 => "spurious interrupt",
        0x64..=0x7C => "autovector interrupt",
        0x80..=0xBC => "TRAP #n",
        _ => "other",
    }
}

/// Writes the last `last` history entries, pipe-separated like Snow's GUI export.
fn write_history(path: &Path, history: &[HistoryEntry], last: usize) -> Result<()> {
    let mut f = File::create(path)?;
    writeln!(
        f,
        "PC|Raw|Cycles|Instruction|D0|D1|D2|D3|D4|D5|D6|D7|A0|A1|A2|A3|A4|A5|A6|A7|SR"
    )?;
    for entry in &history[history.len().saturating_sub(last)..] {
        match entry {
            HistoryEntry::Instruction(HistoryEntryInstruction {
                pc,
                raw,
                cycles,
                initial_regs,
                ..
            }) => {
                let r = initial_regs.clone().unwrap_or_default();
                let mut bytes = raw.iter().copied();
                let dis = Disassembler::from(&mut bytes, *pc)
                    .next()
                    .map_or_else(|| "<invalid>".to_string(), |e| e.str);
                let hex: String = raw.iter().map(|b| format!("{b:02X}")).collect();
                let d: Vec<String> = (0..8).map(|i| format!("{:08X}", r.read_d::<Long>(i))).collect();
                let a: Vec<String> = (0..8).map(|i| format!("{:08X}", r.read_a::<Long>(i))).collect();
                writeln!(
                    f,
                    "{pc:08X}|{hex}|{cycles}|{dis}|{}|{}|{:04X}",
                    d.join("|"),
                    a.join("|"),
                    r.sr.0,
                )?;
            }
            HistoryEntry::Exception { vector, cycles } => writeln!(
                f,
                "--- exception: {} (vector ${vector:03X}) at cycle {cycles}",
                vector_name(*vector)
            )?,
            HistoryEntry::Pagefault { address, write } => writeln!(
                f,
                "--- MMU page fault {address:08X} ({})",
                if *write { "write" } else { "read" }
            )?,
        }
    }
    Ok(())
}

/// Writes the last `last` A-line traps.
fn write_traps(path: &Path, traps: &[SystrapHistoryEntry], last: usize) -> Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "PC|Trap|Cycles")?;
    for t in &traps[traps.len().saturating_sub(last)..] {
        writeln!(f, "{:08X}|{:04X}|{}", t.pc, t.trap, t.cycles)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .init();
    let args = Args::parse();

    let rom = fs::read(&args.rom).with_context(|| format!("reading {}", args.rom.display()))?;
    let model = match args.model {
        Some(m) => m,
        None => MacModel::detect_from_rom(&rom)
            .context("cannot detect model from ROM; pass --model")?,
    };
    let date = chrono::NaiveDateTime::parse_from_str(&args.date, "%Y-%m-%d %H:%M:%S")
        .context("--date must be \"YYYY-MM-DD HH:MM:SS\"")?;

    // Key script: (tick, pressed, scancode), sorted by time.
    let mut key_script: Vec<(Ticks, bool, u8)> = Vec::new();
    for spec in &args.keys {
        let (secs, code, hold) = parse_key(spec)?;
        key_script.push(((secs * SE_CLOCK_HZ) as Ticks, true, code));
        key_script.push((((secs + hold) * SE_CLOCK_HZ) as Ticks, false, code));
    }
    key_script.sort_by_key(|e| e.0);
    let mut key_script = key_script.into_iter().peekable();

    fs::create_dir_all(&args.out_dir)?;

    let (mut emulator, frame_recv) = Emulator::new_with_extra(
        &rom,
        &[],
        model,
        None,
        MouseMode::Disabled,
        args.ram_mb.map(|mb| mb * 1024 * 1024),
        None,
        args.pmmu,
        None,
    )?;
    emulator.set_datetime(date);
    if let Some(disk) = &args.scsi {
        emulator.load_hdd_image(disk, args.scsi_id)?;
        info!("SCSI {}: {}", args.scsi_id, disk.display());
    }

    let cmd = emulator.create_cmd_sender();
    let event_recv = emulator.create_event_recv();
    if args.history > 0 {
        cmd.send(EmulatorCommand::SetInstructionHistory(true))?;
    }
    if args.traps > 0 {
        cmd.send(EmulatorCommand::SetSystrapHistory(true))?;
    }
    for v in &args.break_vectors {
        let vector = u32::from_str_radix(v.trim_start_matches("0x").trim_start_matches('$'), 16)
            .with_context(|| format!("--break-vector '{v}': expected a hex vector address"))?;
        cmd.send(EmulatorCommand::ToggleBreakpoint(Breakpoint::ExceptionVector(vector)))?;
        info!("breakpoint on exception vector ${vector:03X} ({})", vector_name(vector));
    }
    cmd.send(EmulatorCommand::Run)?;
    cmd.send(EmulatorCommand::SetSpeed(EmulatorSpeed::Uncapped))?;

    let total = (args.seconds * SE_CLOCK_HZ) as Ticks;
    let shot_ticks = (args.shot_every * SE_CLOCK_HZ) as Ticks;
    let mut next_shot = shot_ticks;
    let mut latest: Option<(u16, u16, Vec<u8>)> = None;
    let mut obs = Observed::default();

    info!("Running {} for {:.1} emulated seconds", model, args.seconds);
    let start = Instant::now();
    while emulator.get_cycles() < total {
        // Take the frame in its own statement so the lock guard is dropped before ticking.
        let frame = frame_recv.lock().unwrap().take();
        if let Some(buf) = frame {
            let (w, h) = (buf.width(), buf.height());
            latest = Some((w, h, buf.into_inner()));
        }
        if drain_events(&event_recv, &mut obs) {
            warn!(
                "emulator stopped at t={:.2}s — {}",
                emulator.get_cycles() as f64 / SE_CLOCK_HZ,
                regs_summary(obs.regs.as_ref())
            );
            break;
        }

        let now = emulator.get_cycles();
        while let Some((t, pressed, code)) = key_script.next_if(|e| e.0 <= now) {
            info!(
                "t={:.2}s key {} 0x{code:02X}",
                t as f64 / SE_CLOCK_HZ,
                if pressed { "down" } else { "up" }
            );
            let ev = if pressed {
                KeyEvent::KeyDown(code, Keymap::Universal)
            } else {
                KeyEvent::KeyUp(code, Keymap::Universal)
            };
            cmd.send(EmulatorCommand::KeyEvent(ev))?;
        }

        emulator.tick(1, ())?;

        let cycles = emulator.get_cycles();
        if shot_ticks > 0 && cycles >= next_shot {
            if let Some((w, h, rgba)) = &latest {
                let t = cycles as f64 / SE_CLOCK_HZ;
                let path = args.out_dir.join(format!("shot-{t:06.1}s.png"));
                write_png(&path, *w, *h, rgba)?;
                info!("t={t:.1}s {} -> {}", regs_summary(obs.regs.as_ref()), path.display());
            }
            next_shot += shot_ticks;
        }
    }
    drain_events(&event_recv, &mut obs);

    let secs = start.elapsed().as_secs_f64();
    let emulated = emulator.get_cycles() as f64 / SE_CLOCK_HZ;
    info!(
        "{emulated:.1} emulated s in {secs:.1} host s ({:.2}x real time)",
        emulated / secs
    );
    info!("final: {}", regs_summary(obs.regs.as_ref()));
    if let Some((w, h, rgba)) = &latest {
        let path = args.out_dir.join("final.png");
        write_png(&path, *w, *h, rgba)?;
        info!("final -> {}", path.display());
    }
    if let Some(path) = &args.dump_ram {
        fs::write(path, &obs.ram)?;
        info!("RAM ({} bytes) -> {}", obs.ram.len(), path.display());
    }
    if args.history > 0 {
        let path = args.out_dir.join("history.txt");
        write_history(&path, &obs.history, args.history)?;
        info!("history ({} of {} entries) -> {}", args.history.min(obs.history.len()), obs.history.len(), path.display());
    }
    if args.traps > 0 {
        let path = args.out_dir.join("traps.txt");
        write_traps(&path, &obs.traps, args.traps)?;
        info!("traps ({} of {}) -> {}", args.traps.min(obs.traps.len()), obs.traps.len(), path.display());
    }
    Ok(())
}
