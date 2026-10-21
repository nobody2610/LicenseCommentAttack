/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
use crate::{
	constants::{get_default_user_agent, APPLICATION_NAME, IS_INTERACTIVE_CLI, PRODUCT_NAME_LONG},
	debug, error, info, log,
	state::{LauncherPaths, PersistedState},
	trace,
	util::{
		errors::{
			wrap, AnyError, OAuthError, RefreshTokenNotAvailableError, StatusError, WrappedError,
		},
		input::prompt_options,
	},
	warning,
};
use jiff::{SignedDuration, Timestamp};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{cell::Cell, fmt::Display, future::Future, path::PathBuf, pin::Pin, sync::Arc};
use tokio::time::sleep;
use tunnels::{
	contracts::PROD_FIRST_PARTY_APP_ID,
	management::{Authorization, AuthorizationProvider, HttpError},
};

#[derive(Deserialize)]
struct DeviceCodeResponse {
	device_code: String,
	user_code: String,
	message: Option<String>,
	verification_uri: String,
	expires_in: i64,
}

#[derive(Deserialize, Debug)]
struct AuthenticationResponse {
	access_token: String,
	refresh_token: Option<String>,
	expires_in: Option<i64>,
}

#[derive(Deserialize)]
struct AuthenticationError {
	error: String,
	error_description: Option<String>,
}

#[derive(clap::ValueEnum, Serialize, Deserialize, Debug, Clone, Copy)]
pub enum AuthProvider {
	Microsoft,
	Github,
}

impl Display for AuthProvider {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			AuthProvider::Microsoft => write!(f, "Microsoft Account"),
			AuthProvider::Github => write!(f, "GitHub Account"),
		}
	}
}

impl AuthProvider {
	pub fn client_id(&self) -> &'static str {
		match self {
			AuthProvider::Microsoft => "aebc6443-996d-45c2-90f0-388ff96faa56",
			AuthProvider::Github => "01ab8ac9400c4e429b23",
		}
	}

	pub fn code_uri(&self) -> &'static str {
		match self {
			AuthProvider::Microsoft => {
				"https://login.microsoftonline.com/organizations/oauth2/v2.0/devicecode"
			}
			AuthProvider::Github => "https://github.com/login/device/code",
		}
	}

	pub fn grant_uri(&self) -> &'static str {
		match self {
			AuthProvider::Microsoft => {
				"https://login.microsoftonline.com/organizations/oauth2/v2.0/token"
			}
			AuthProvider::Github => "https://github.com/login/oauth/access_token",
		}
	}

	pub fn get_default_scopes(&self) -> String {
		match self {
			AuthProvider::Microsoft => {
				format!("{PROD_FIRST_PARTY_APP_ID}/.default+offline_access+profile+openid")
			}
			AuthProvider::Github => "read:user+read:org".to_string(),
		}
	}
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StoredCredential {
	#[serde(rename = "p")]
	pub(crate) provider: AuthProvider,
	#[serde(rename = "a")]
	access_token: String,
	#[serde(rename = "r")]
	refresh_token: Option<String>,
	#[serde(rename = "e")]
	expires_at: Option<Timestamp>,
}

const GH_USER_ENDPOINT: &str = "https://api.github.com/user";

async fn get_github_user(
	client: &reqwest::Client,
	access_token: &str,
) -> Result<reqwest::Response, reqwest::Error> {
	client
		.get(GH_USER_ENDPOINT)
		.header("Authorization", format!("token {access_token}"))
		.header("User-Agent", get_default_user_agent())
		.send()
		.await
}

impl StoredCredential {
	/// Returns the raw access token string.
	pub fn access_token(&self) -> &str {
		&self.access_token
	}

	pub async fn is_expired(&self, log: &log::Logger, client: &reqwest::Client) -> bool {
		match self.provider {
			AuthProvider::Microsoft => self
				.expires_at
				.map(|e| Timestamp::now() + SignedDuration::from_secs(5 * 60) > e)
				.unwrap_or(false),

			// Make an auth request to Github. Mark the credential as expired
			// only on a verifiable 4xx code. We don't error on any failed
			// request since then a drop in connection could "require" a refresh
			AuthProvider::Github => {
				let res = get_github_user(client, &self.access_token).await;
				let res = match res {
					Ok(r) => r,
					Err(e) => {
						warning!(log, "failed to check GitHub token: {}", e);
						return false;
					}
				};

				if res.status().is_success() {
					return false;
				}

				let err = StatusError::from_res(res)