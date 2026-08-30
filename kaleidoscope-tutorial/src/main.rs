use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};

use kaleidoscope_tutorial::{evaluate, lex, parse_program};

#[derive(Clone, Copy)]
enum Mode {
    Ast,
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
        } else if argument == "--ast" {
            mode = Mode::Ast;
        } else if path.replace(argument).is_some() {
            return Err("expected at most one source path".into());
        }
    }

    let source = read_source(path)?;
    match mode {
        Mode::Ast => println!("{:#?}", parse_program(&source)?),
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
