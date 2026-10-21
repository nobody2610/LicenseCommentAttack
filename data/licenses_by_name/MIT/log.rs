/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
use jiff::Zoned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::{
	io::Write,
	sync::atomic::{AtomicU32, Ordering},
};
use std::{path::Path, sync::Arc};

use crate::constants::COLORS_ENABLED;

static INSTANCE_COUNTER: AtomicU32 = AtomicU32::new(0);

// Gets a next incrementing number that can be used in logs
pub fn next_counter() -> u32 {
	INSTANCE_COUNTER.fetch_add(1, Ordering::SeqCst)
}

// Log level
#[derive(
	clap::ValueEnum, PartialEq, Eq, PartialOrd, Clone, Copy, Debug, Serialize, Deserialize, Default,
)]
pub enum Level {
	Trace = 0,
	Debug,
	#[default]
	Info,
	Warn,
	Error,
	Critical,
	Off,
}

impl fmt::Display for Level {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Level::Critical => write!(f, "critical"),
			Level::Debug => write!(f, "debug"),
			Level::Error => write!(f, "error"),
			Level::Info => write!(f, "info"),
			Level::Off => write!(f, "off"),
			Level::Trace => write!(f, "trace"),
			Level::Warn => write!(f, "warn"),
		}
	}
}

impl Level {
	pub fn name(&self) -> Option<&str> {
		match self {
			Level::Trace => Some("trace"),
			Level::Debug => Some("debug"),
			Level::Info => Some("info"),
			Level::Warn => Some("warn"),
			Level::Error => Some("error"),
			Level::Critical => Some("critical"),
			Level::Off => None,
		}
	}

	pub fn color_code(&self) -> Option<&str> {
		if !*COLORS_ENABLED {
			return None;
		}

		match self {
			Level::Trace => None,
			Level::Debug => Some("\x1b[36m"),
			Level::Info => Some("\x1b[35m"),
			Level::Warn => Some("\x1b[33m"),
			Level::Error => Some("\x1b[31m"),
			Level::Critical => Some("\x1b[31m"),
			Level::Off => None,
		}
	}

	pub fn to_u8(self) -> u8 {
		self as u8
	}
}

pub fn new_tunnel_prefix() -> String {
	format!("[tunnel.{}]", next_counter())
}

pub fn new_code_server_prefix() -> String {
	format!("[codeserver.{}]", next_counter())
}

pub fn new_rpc_prefix() -> String {
	format!("[rpc.{}]", next_counter())
}

// Base logger implementation
#[derive(Clone)]
pub struct Logger {
	sink: Vec<Box<dyn LogSink>>,
	prefix: Option<String>,
}

// Copy trick from https://stackoverflow.com/a/30353928
pub trait LogSinkClone {
	fn clone_box(&self) -> Box<dyn LogSink>;
}

impl<T> LogSinkClone for T
where
	T: 'static + LogSink + Clone,
{
	fn clone_box(&self) -> Box<dyn LogSink> {
		Box::new(self.clone())
	}
}

pub trait LogSink: LogSinkClone + Sync + Send {
	fn write_log(&self, level: Level, prefix: &str, message: &str);
	fn write_result(&self, message: &str);
}

impl Clone for Box<dyn LogSink> {
	fn clone(&self) -> Box<dyn LogSink> {
		self.clone_box()
	}
}

/// The basic log sink that writes output to stdout, with colors when relevant.
#[derive(Clone)]
pub struct StdioLogSink {
	level: Level,
}

impl LogSink for StdioLogSink {
	fn write_log(&self, level: Level, prefix: &str, message: &str) {
		if level < self.level {
			return;
		}

		emit(level, prefix, message);
	}

	fn write_result(&self, message: &str) {
		println!("{message}");
	}
}

#[derive(Clone)]
pub struct FileLogSink {
	level: Level,
	file: Arc<std::sync::Mutex<std::fs::File>>,
}

const FILE_LOG_SIZE_LIMIT: u64 = 1024 * 1024 * 10; // 10MB

impl FileLogSink {
	pub fn new(level: Level, path: &Path) -> std::io::Result<Self> {
		// Truncate the service log occasionally to avoid growing infinitely
		if matches!(path.metadata(), Ok(m) if m.len() > FILE_LOG_SIZE_LIMIT) {
			// ignore errors, can happen if another process is writing right now
			let _ = std::fs::remove_file(path);
		}

		let file = std::fs::OpenOptions::new()
			.append(true)
			.create(true)
			.open(path)?;

		Ok(Self {
			level,
			file: Arc::new(std::sync::Mutex::new(file)),
		})
	}
}

impl LogSink for FileLogSink {
	fn write_log(&self, level: Level, prefix: &str, message: &str) {
		if level < self.level {
			return;
		}

		let line = format(level, prefix, message, false);

		// ignore any errors, not much we can do if logging fails...
		self.file.lock().unwrap().write_all(line.as_bytes()).ok();
	}

	fn write_resu