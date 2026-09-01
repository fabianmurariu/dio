use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};

use kaleidoscope_tutorial::{Session, evaluate, lex, parse_program};

#[derive(Clone, Copy)]
enum Mode {
    Ast,
    Repl,
    Run,
    Tokens,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kaleidoscope: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut mode = Mode::Ast;
    let mut path = None;
    for argument in env::args_os().skip(1) {
        if argument == "--tokens" {
            mode = Mode::Tokens;
        } else if argument == "--run" {
            mode = Mode::Run;
        } else if argument == "--repl" {
            mode = Mode::Repl;
        } else if argument == "--ast" {
            mode = Mode::Ast;
        } else if path.replace(argument).is_some() {
            return Err("expected at most one source path".into());
        }
    }

    if matches!(mode, Mode::Repl) {
        return run_repl(path);
    }

    let source = read_source(path)?;
    match mode {
        Mode::Ast => println!("{:#?}", parse_program(&source)?),
        Mode::Repl => unreachable!("REPL mode returns before reading a complete source"),
        Mode::Run => {
            let program = parse_program(&source)?;
            for value in evaluate(&program)? {
                println!("Evaluated to {value:.6}");
            }
        }
        Mode::Tokens => {
            for token in lex(&source)? {
                println!("{token}");
            }
        }
    }
    Ok(())
}

fn run_repl(path: Option<OsString>) -> Result<(), Box<dyn Error>> {
    match path {
        Some(path) => run_repl_reader(BufReader::new(fs::File::open(path)?), false),
        None => {
            let stdin = io::stdin();
            let interactive = stdin.is_terminal();
            run_repl_reader(stdin.lock(), interactive)
        }
    }
}

fn run_repl_reader(mut reader: impl BufRead, interactive: bool) -> Result<(), Box<dyn Error>> {
    let mut session = Session::new();
    let mut submission = String::new();

    loop {
        if interactive {
            print!("ready> ");
            io::stdout().flush()?;
        }
        submission.clear();
        if reader.read_line(&mut submission)? == 0 {
            if interactive {
                println!();
            }
            return Ok(());
        }
        if submission.trim().is_empty() {
            continue;
        }

        match session.submit_source(&submission) {
            Ok(values) => {
                for value in values {
                    println!("Evaluated to {value:.6}");
                }
            }
            Err(error) => eprintln!("kaleidoscope: {error}"),
        }
    }
}

fn read_source(path: Option<OsString>) -> io::Result<String> {
    match path {
        Some(path) => fs::read_to_string(path),
        None => {
            let mut source = String::new();
            io::stdin().read_to_string(&mut source)?;
            Ok(source)
        }
    }
}
