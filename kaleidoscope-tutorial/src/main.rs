use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};

use kaleidoscope_tutorial::lex;

fn main() {
    if let Err(error) = run() {
        eprintln!("kaleidoscope: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let path = arguments.next();
    if let Some(unexpected) = arguments.next() {
        return Err(format!(
            "expected at most one source path, found unexpected argument {:?}",
            unexpected
        )
        .into());
    }

    let source = read_source(path)?;
    for token in lex(&source)? {
        println!("{token}");
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
