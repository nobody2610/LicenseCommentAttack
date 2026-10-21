/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
use std::{
	collections::HashMap,
	future,
	sync::{
		atomic::{AtomicU32, Ordering},
		Arc, Mutex,
	},
};

use crate::log;
use futures::{future::BoxFuture, Future, FutureExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::{
	io::{AsyncReadExt, AsyncWriteExt, DuplexStream, WriteHalf},
	sync::{mpsc, oneshot},
};

use crate::util::errors::AnyError;

pub type SyncMethod = Arc<dyn Send + Sync + Fn(Option<u32>, &[u8]) -> Option<Vec<u8>>>;
pub type AsyncMethod =
	Arc<dyn Send + Sync + Fn(Option<u32>, &[u8]) -> BoxFuture<'static, Option<Vec<u8>>>>;
pub type Duplex = Arc<
	dyn Send
		+ Sync
		+ Fn(Option<u32>, &[u8]) -> (Option<StreamDto>, BoxFuture<'static, Option<Vec<u8>>>),
>;

pub enum Method {
	Sync(SyncMethod),
	Async(AsyncMethod),
	Duplex(Duplex),
}

/// Serialization is given to the RpcBuilder and defines how data gets serialized
/// when callinth methods.
pub trait Serialization: Send + Sync + 'static {
	fn serialize(&self, value: impl Serialize) -> Vec<u8>;
	fn deserialize<P: DeserializeOwned>(&self, b: &[u8]) -> Result<P, AnyError>;
}

/// RPC is a basic, transport-agnostic builder for RPC methods. You can
/// register methods to it, then call `.build()` to get a "dispatcher" type.
pub struct RpcBuilder<S> {
	serializer: Arc<S>,
	methods: HashMap<&'static str, Method>,
	calls: Arc<Mutex<HashMap<u32, DispatchMethod>>>,
}

impl<S: Serialization> RpcBuilder<S> {
	/// Creates a new empty RPC builder.
	pub fn new(serializer: S) -> Self {
		Self {
			serializer: Arc::new(serializer),
			methods: HashMap::new(),
			calls: Arc::new(std::sync::Mutex::new(HashMap::new())),
		}
	}

	/// Creates a caller that will be connected to any eventual dispatchers,
	/// and that sends data to the "tx" channel.
	pub fn get_caller(&mut self, sender: mpsc::UnboundedSender<Vec<u8>>) -> RpcCaller<S> {
		RpcCaller {
			serializer: self.serializer.clone(),
			calls: self.calls.clone(),
			sender,
		}
	}

	/// Gets a method builder.
	pub fn methods<C: Send + Sync + 'static>(self, context: C) -> RpcMethodBuilder<S, C> {
		RpcMethodBuilder {
			context: Arc::new(context),
			serializer: self.serializer,
			methods: self.methods,
			calls: self.calls,
		}
	}
}

pub struct RpcMethodBuilder<S, C> {
	context: Arc<C>,
	serializer: Arc<S>,
	methods: HashMap<&'static str, Method>,
	calls: Arc<Mutex<HashMap<u32, DispatchMethod>>>,
}

#[derive(Serialize)]
struct DuplexStreamStarted {
	pub for_request_id: u32,
	pub stream_ids: Vec<u32>,
}

impl<S: Serialization, C: Send + Sync + 'static> RpcMethodBuilder<S, C> {
	/// Registers a synchronous rpc call that returns its result directly.
	pub fn register_sync<P, R, F>(&mut self, method_name: &'static str, callback: F)
	where
		P: DeserializeOwned,
		R: Serialize,
		F: Fn(P, &C) -> Result<R, AnyError> + Send + Sync + 'static,
	{
		if self.methods.contains_key(method_name) {
			panic!("Method already registered: {method_name}");
		}

		let serial = self.serializer.clone();
		let context = self.context.clone();
		self.methods.insert(
			method_name,
			Method::Sync(Arc::new(move |id, body| {
				let param = match serial.deserialize::<RequestParams<P>>(body) {
					Ok(p) => p,
					Err(err) => {
						return id.map(|id| {
							serial.serialize(ErrorResponse {
								id,
								error: ResponseError {
									code: 0,
									message: format!("{err:?}"),
								},
							})
						})
					}
				};

				match callback(param.params, &context) {
					Ok(result) => id.map(|id| serial.serialize(&SuccessResponse { id, result })),
					Err(err) => id.map(|id| {
						serial.serialize(ErrorResponse {
							id,
							error: ResponseError {
								code: -1,
								message: format!("{err:?}"),
							},
						})
					}),
				}
			})),
		);
	}

	/// Registers an async rpc call that returns a Future.
	pub fn register_async<P, R, Fut, F>(&mut self, method_name: &'static str, callback: F)
	where
		P: DeserializeOwned + Send + 'static,
		R: Serialize + Send + Sync + 'static,
		Fut: Future<Output = Result<R, A