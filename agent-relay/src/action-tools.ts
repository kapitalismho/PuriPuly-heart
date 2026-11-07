export const actionTools = [
  {
    "type": "function",
    "name": "get_app_context",
    "description": "Read app_context modules omitted from the initial context or refresh stale facts. A read returns core plus the same-name topic module; overview returns all. An omitted module is not an absent or disabled feature. An explicit accounts read also stages user-click account actions; initial facts alone do not show controls. No identities, secrets or live balance refresh. Diagnostics accepts observed diagnostic_id/operation_id or next_after, returning at most 10 records in core.diagnostics; missing/evicted records do not prove recovery. Other topics require null filters. No raw logs or runtime changes.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "topic": {
          "type": "string",
          "enum": [
            "accounts",
            "audio",
            "diagnostics",
            "general",
            "languages",
            "output",
            "overview",
            "recognition",
            "translation"
          ]
        },
        "diagnostic_id": {
          "type": [
            "string",
            "null"
          ],
          "minLength": 1,
          "maxLength": 64
        },
        "operation_id": {
          "type": [
            "string",
            "null"
          ],
          "minLength": 1,
          "maxLength": 64
        },
        "after": {
          "type": [
            "integer",
            "null"
          ],
          "minimum": 0
        }
      },
      "required": [
        "topic",
        "diagnostic_id",
        "operation_id",
        "after"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "change_settings",
    "description": "Propose supported settings changes within the user's request. Every change requires a user-approved card; this tool never applies settings directly. Null means leave unchanged. Use actual settings choices; submit translation model and connection together. The app queues cards with at most two setting blocks, keeping coupled settings together, and returns needs_confirmation. Explain the proposal and end the turn; do not resubmit queued changes. Each user approval applies only that card before the next appears. Do not retry a revision conflict without reconsidering current state.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "changes": {
          "type": "object",
          "properties": {
            "audio.input_device": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ]
            },
            "audio.input_host_api": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ]
            },
            "audio.output_device": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ]
            },
            "chatbox.activation_notice.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "chatbox.source.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "clipboard.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "languages": {
              "anyOf": [
                {
                  "type": "object",
                  "properties": {
                    "source": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "ar",
                            "bg",
                            "ca",
                            "zh-CN",
                            "zh-TW",
                            "cs",
                            "da",
                            "nl",
                            "en",
                            "et",
                            "fi",
                            "fr",
                            "de",
                            "el",
                            "hi",
                            "hu",
                            "id",
                            "it",
                            "ja",
                            "ko",
                            "lv",
                            "lt",
                            "ms",
                            "no",
                            "pl",
                            "pt",
                            "ro",
                            "ru",
                            "sk",
                            "es",
                            "sv",
                            "th",
                            "tr",
                            "uk",
                            "vi"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "target": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "ar",
                            "bg",
                            "ca",
                            "zh-CN",
                            "zh-TW",
                            "cs",
                            "da",
                            "nl",
                            "en",
                            "et",
                            "fi",
                            "fr",
                            "de",
                            "el",
                            "hi",
                            "hu",
                            "id",
                            "it",
                            "ja",
                            "ko",
                            "lv",
                            "lt",
                            "ms",
                            "no",
                            "pl",
                            "pt",
                            "ro",
                            "ru",
                            "sk",
                            "es",
                            "sv",
                            "th",
                            "tr",
                            "uk",
                            "vi"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "secondary_target": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "ar",
                            "bg",
                            "ca",
                            "zh-CN",
                            "zh-TW",
                            "cs",
                            "da",
                            "nl",
                            "en",
                            "et",
                            "fi",
                            "fr",
                            "de",
                            "el",
                            "hi",
                            "hu",
                            "id",
                            "it",
                            "ja",
                            "ko",
                            "lv",
                            "lt",
                            "ms",
                            "no",
                            "pl",
                            "pt",
                            "ro",
                            "ru",
                            "sk",
                            "es",
                            "sv",
                            "th",
                            "tr",
                            "uk",
                            "vi",
                            ""
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "peer_source": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "ar",
                            "bg",
                            "ca",
                            "zh-CN",
                            "zh-TW",
                            "cs",
                            "da",
                            "nl",
                            "en",
                            "et",
                            "fi",
                            "fr",
                            "de",
                            "el",
                            "hi",
                            "hu",
                            "id",
                            "it",
                            "ja",
                            "ko",
                            "lv",
                            "lt",
                            "ms",
                            "no",
                            "pl",
                            "pt",
                            "ro",
                            "ru",
                            "sk",
                            "es",
                            "sv",
                            "th",
                            "tr",
                            "uk",
                            "vi",
                            ""
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "peer_target": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "ar",
                            "bg",
                            "ca",
                            "zh-CN",
                            "zh-TW",
                            "cs",
                            "da",
                            "nl",
                            "en",
                            "et",
                            "fi",
                            "fr",
                            "de",
                            "el",
                            "hi",
                            "hu",
                            "id",
                            "it",
                            "ja",
                            "ko",
                            "lv",
                            "lt",
                            "ms",
                            "no",
                            "pl",
                            "pt",
                            "ro",
                            "ru",
                            "sk",
                            "es",
                            "sv",
                            "th",
                            "tr",
                            "uk",
                            "vi",
                            ""
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "peer_source_mode": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "manual",
                            "auto"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    }
                  },
                  "required": [
                    "source",
                    "target",
                    "secondary_target",
                    "peer_source",
                    "peer_target",
                    "peer_source_mode"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "null"
                }
              ]
            },
            "locale": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "ko",
                    "en",
                    "ja"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "osc.connection": {
              "anyOf": [
                {
                  "type": "object",
                  "properties": {
                    "mode": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "automatic",
                            "manual",
                            "off"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "send_port": {
                      "anyOf": [
                        {
                          "type": "integer",
                          "minimum": 1,
                          "maximum": 65535
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "receive_port": {
                      "anyOf": [
                        {
                          "type": "integer",
                          "minimum": 1,
                          "maximum": 65535
                        },
                        {
                          "type": "null"
                        }
                      ]
                    }
                  },
                  "required": [
                    "mode",
                    "send_port",
                    "receive_port"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "null"
                }
              ]
            },
            "overlay.background_alpha": {
              "anyOf": [
                {
                  "type": "number",
                  "minimum": 0,
                  "maximum": 1
                },
                {
                  "type": "null"
                }
              ]
            },
            "overlay.desktop_size": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "tiny",
                    "xsmall",
                    "small",
                    "medium",
                    "large",
                    "xlarge"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "overlay.peer_original.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "overlay.swap_caption_languages": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "overlay.target": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "steamvr",
                    "desktop"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "overlay.translation.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "peer_expected_languages": {
              "anyOf": [
                {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "enum": [
                      "ar",
                      "bg",
                      "ca",
                      "zh-CN",
                      "zh-TW",
                      "cs",
                      "da",
                      "nl",
                      "en",
                      "et",
                      "fi",
                      "fr",
                      "de",
                      "el",
                      "hi",
                      "hu",
                      "id",
                      "it",
                      "ja",
                      "ko",
                      "lv",
                      "lt",
                      "ms",
                      "no",
                      "pl",
                      "pt",
                      "ro",
                      "ru",
                      "sk",
                      "es",
                      "sv",
                      "th",
                      "tr",
                      "uk",
                      "vi"
                    ]
                  }
                },
                {
                  "type": "null"
                }
              ]
            },
            "peer_stt.provider": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "local_cpu_auto",
                    "local_granite",
                    "local_gigaam",
                    "local_fun_asr_nano",
                    "local_fun_asr_mlt_nano",
                    "local_parakeet_ja",
                    "local_qwen_gpu",
                    "deepgram",
                    "gemini_transcribe",
                    "elevenlabs_scribe",
                    "qwen_audio",
                    "soniox",
                    "rolling_free",
                    "custom",
                    "custom_offline",
                    "custom_realtime"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "peer_vad.hangover_ms": {
              "anyOf": [
                {
                  "type": "integer",
                  "minimum": 0
                },
                {
                  "type": "null"
                }
              ]
            },
            "peer_vad.pre_roll_ms": {
              "anyOf": [
                {
                  "type": "integer",
                  "minimum": 0
                },
                {
                  "type": "null"
                }
              ]
            },
            "peer_vad.speech_threshold": {
              "anyOf": [
                {
                  "type": "number",
                  "minimum": 0.1,
                  "maximum": 1
                },
                {
                  "type": "null"
                }
              ]
            },
            "self_vad.speech_threshold": {
              "anyOf": [
                {
                  "type": "number",
                  "minimum": 0.1,
                  "maximum": 1
                },
                {
                  "type": "null"
                }
              ]
            },
            "stt.cloud_free_tier_providers": {
              "anyOf": [
                {
                  "type": "array",
                  "items": {
                    "type": "string",
                    "enum": [
                      "elevenlabs_scribe",
                      "gemini_transcribe",
                      "deepgram"
                    ]
                  },
                  "minItems": 1
                },
                {
                  "type": "null"
                }
              ]
            },
            "stt.gpu_device_id": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ]
            },
            "stt.provider": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "local_cpu_auto",
                    "local_granite",
                    "local_gigaam",
                    "local_fun_asr_nano",
                    "local_fun_asr_mlt_nano",
                    "local_parakeet_ja",
                    "local_qwen_gpu",
                    "deepgram",
                    "gemini_transcribe",
                    "elevenlabs_scribe",
                    "qwen_audio",
                    "soniox",
                    "rolling_free",
                    "custom",
                    "custom_offline",
                    "custom_realtime"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "stt.soniox.enable_speaker_diarization": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "telemetry.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.connection": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "public_api",
                    "managed",
                    "openrouter",
                    "managed_china",
                    "official_byok",
                    "chatgpt",
                    "cpu",
                    "gpu",
                    "ollama",
                    "custom_http"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.connection_history": {
              "anyOf": [
                {
                  "type": "object",
                  "properties": {
                    "index_translate_35b": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "public_api"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "gemma4_26b_31b": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "managed",
                            "openrouter"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "deepseek_v4_flash": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "openrouter"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "deepseek_v4_flash_41": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "managed",
                            "managed_china",
                            "openrouter",
                            "official_byok"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "gpt_6_luna": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "chatgpt",
                            "openrouter",
                            "official_byok"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "qwen38_flash": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "official_byok"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "managed_gemma": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "cpu",
                            "gpu"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "managed_gemma_26b": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "gpu"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "local_llm": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "ollama"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    },
                    "custom_http": {
                      "anyOf": [
                        {
                          "type": "string",
                          "enum": [
                            "custom_http"
                          ]
                        },
                        {
                          "type": "null"
                        }
                      ]
                    }
                  },
                  "required": [
                    "index_translate_35b",
                    "gemma4_26b_31b",
                    "deepseek_v4_flash",
                    "deepseek_v4_flash_41",
                    "gpt_6_luna",
                    "qwen38_flash",
                    "managed_gemma",
                    "managed_gemma_26b",
                    "local_llm",
                    "custom_http"
                  ],
                  "additionalProperties": false
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.gpu_device_id": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.http_extension_id": {
              "anyOf": [
                {
                  "type": "string"
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.model": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "index_translate_35b",
                    "gemma4_26b_31b",
                    "deepseek_v4_flash",
                    "deepseek_v4_flash_41",
                    "gpt_6_luna",
                    "qwen38_flash",
                    "managed_gemma",
                    "managed_gemma_26b",
                    "local_llm",
                    "custom_http"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.previous_llm_model": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "index_translate_35b",
                    "gemma4_26b_31b",
                    "deepseek_v4_flash",
                    "deepseek_v4_flash_41",
                    "gpt_6_luna",
                    "qwen38_flash",
                    "managed_gemma",
                    "managed_gemma_26b",
                    "local_llm",
                    "custom_http"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "translation.qwen.region": {
              "anyOf": [
                {
                  "type": "string",
                  "enum": [
                    "beijing",
                    "singapore"
                  ]
                },
                {
                  "type": "null"
                }
              ]
            },
            "vrc_mic_intercept.enabled": {
              "anyOf": [
                {
                  "type": "boolean"
                },
                {
                  "type": "null"
                }
              ]
            }
          },
          "required": [
            "audio.input_device",
            "audio.input_host_api",
            "audio.output_device",
            "chatbox.activation_notice.enabled",
            "chatbox.source.enabled",
            "clipboard.enabled",
            "languages",
            "locale",
            "osc.connection",
            "overlay.background_alpha",
            "overlay.desktop_size",
            "overlay.peer_original.enabled",
            "overlay.swap_caption_languages",
            "overlay.target",
            "overlay.translation.enabled",
            "peer_expected_languages",
            "peer_stt.provider",
            "peer_vad.hangover_ms",
            "peer_vad.pre_roll_ms",
            "peer_vad.speech_threshold",
            "self_vad.speech_threshold",
            "stt.cloud_free_tier_providers",
            "stt.gpu_device_id",
            "stt.provider",
            "stt.soniox.enable_speaker_diarization",
            "telemetry.enabled",
            "translation.connection",
            "translation.connection_history",
            "translation.gpu_device_id",
            "translation.http_extension_id",
            "translation.model",
            "translation.previous_llm_model",
            "translation.qwen.region",
            "vrc_mic_intercept.enabled"
          ],
          "additionalProperties": false
        }
      },
      "required": [
        "changes"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "control_app",
    "description": "Read current-instance status or run a few non-settings app actions. query.* returns projected metadata only, never private settings, raw content or secret values. overlay.set, audio.retry and models.retry run immediately; inspect the terminal receipt. secrets.delete, auth.logout and overlay.position.reset only create a user-approval card like change_settings; explain the effect and end the turn. Deleting a stored key does not revoke it at the provider. Model downloads or imports, key entry, account login, manual text, restart and overlay calibration are not available here; point the user to the existing app screen.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "request": {
          "anyOf": [
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "audio.retry"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "auth.logout"
                  ]
                },
                "provider": {
                  "type": "string",
                  "enum": [
                    "qq",
                    "discord",
                    "openrouter",
                    "chatgpt"
                  ]
                }
              },
              "required": [
                "action",
                "provider"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "models.retry"
                  ]
                },
                "backend": {
                  "anyOf": [
                    {
                      "type": "string",
                      "enum": [
                        "gpu"
                      ]
                    },
                    {
                      "type": "null"
                    }
                  ]
                }
              },
              "required": [
                "action",
                "backend"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "overlay.position.reset"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "overlay.set"
                  ]
                },
                "enabled": {
                  "type": "boolean"
                }
              },
              "required": [
                "action",
                "enabled"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.accounts.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.app.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.audio.devices"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.auth.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.capture.channels"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.consent.peer_translation"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.diagnostics.health"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.diagnostics.history"
                  ]
                },
                "operation_id": {
                  "anyOf": [
                    {
                      "type": "string",
                      "minLength": 1,
                      "maxLength": 128
                    },
                    {
                      "type": "null"
                    }
                  ]
                },
                "diagnostic_id": {
                  "anyOf": [
                    {
                      "type": "string",
                      "minLength": 1,
                      "maxLength": 128
                    },
                    {
                      "type": "null"
                    }
                  ]
                },
                "after": {
                  "anyOf": [
                    {
                      "type": "integer",
                      "minimum": 0
                    },
                    {
                      "type": "null"
                    }
                  ]
                },
                "limit": {
                  "anyOf": [
                    {
                      "type": "integer",
                      "minimum": 1,
                      "maximum": 100
                    },
                    {
                      "type": "null"
                    }
                  ]
                }
              },
              "required": [
                "action",
                "operation_id",
                "diagnostic_id",
                "after",
                "limit"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.gpu.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.models.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.osc.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.overlay.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.providers.status"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.secrets.presence"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.settings.choices"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "query.settings.current"
                  ]
                }
              },
              "required": [
                "action"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "secrets.delete"
                  ]
                },
                "name": {
                  "type": "string",
                  "enum": [
                    "openrouter_api_key",
                    "openai_api_key",
                    "deepseek_api_key",
                    "deepgram_api_key",
                    "gemini_api_key",
                    "elevenlabs_scribe_api_key",
                    "soniox_api_key",
                    "local_llm_api_key",
                    "custom_stt_api_key",
                    "alibaba_api_key_beijing",
                    "alibaba_api_key_singapore"
                  ]
                }
              },
              "required": [
                "action",
                "name"
              ],
              "additionalProperties": false
            }
          ]
        }
      },
      "required": [
        "request"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "withdraw_confirmation",
    "description": "Withdraw the exact current pending settings card identified by app_context.core.pending_confirmation.id and its remaining unapproved cards when the user cancels that proposal. A question about the card or an unrelated message does not withdraw it. Stale or unknown IDs are rejected. This never approves, applies or undoes settings, and creates no app-notice continuation.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "card_id": {
          "type": "string",
          "minLength": 1
        }
      },
      "required": [
        "card_id"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "set_capture",
    "description": "Set Self or Peer capture through the existing application owner. This cannot accept Peer terms; action_required means the user must complete that existing consent step. Inspect the terminal receipt.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "channel": {
          "type": "string",
          "enum": [
            "self",
            "peer"
          ]
        },
        "enabled": {
          "type": "boolean"
        }
      },
      "required": [
        "channel",
        "enabled"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "set_translation",
    "description": "Set translation through the existing application owner without authorizing accounts or changing providers. Inspect the terminal receipt.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean"
        }
      },
      "required": [
        "enabled"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "diagnose_microphone",
    "description": "Run bounded local microphone capture/VAD diagnosis on the current route and actual enumerated alternatives. No network recognizer or saved settings changes. Inspect candidates, cleanup and restoration in the terminal receipt. Frames or speech activity are not recognition success.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "retry_osc_discovery",
    "description": "Run one bounded automatic OSC discovery requery/recovery using the current connection settings and await its terminal internal application result. Does not enable disabled OSC, change manual mode or saved destinations, or send chatbox text. Target application does not prove VRChat receipt or display.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "get_provider_setup",
    "description": "Read app-owned official provider setup links and the existing secure-field identity. No navigation, browser opening, provider switching or key entry. Returned metadata does not mean a link or key input is currently visible.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "provider": {
          "type": "string",
          "enum": [
            "alibaba_beijing",
            "alibaba_singapore",
            "deepgram",
            "deepseek",
            "elevenlabs_scribe",
            "gemini",
            "openai",
            "openrouter",
            "soniox"
          ]
        }
      },
      "required": [
        "provider"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "open_settings",
    "description": "Open a supported settings section without changing settings. For key help open api with the known provider to expose its existing secure input, otherwise null. This does not select that provider or start browser authorization. Navigation never proves key verification or user-step completion.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {
        "section": {
          "type": "string",
          "enum": [
            "api",
            "general",
            "overlay",
            "prompts"
          ]
        },
        "provider": {
          "type": [
            "string",
            "null"
          ],
          "enum": [
            "alibaba_beijing",
            "alibaba_singapore",
            "deepgram",
            "deepseek",
            "elevenlabs_scribe",
            "gemini",
            "openai",
            "openrouter",
            "soniox",
            null
          ]
        }
      },
      "required": [
        "section",
        "provider"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "probe_self_recognition",
    "description": "Check Self recognition independently with app-owned synthetic Korean speech (about one second) in an isolated session of its configured backend. No arguments, live capture, playback, provider switching, settings changes or transcript/history/translation/output publication. Inspect the terminal receipt. Success means nonempty final synthetic recognition only, not accuracy, other-language quality, microphone/loopback capture or display. The existing configured connection may incur ordinary recognition cost.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "probe_peer_recognition",
    "description": "Check Peer recognition independently with app-owned synthetic Korean speech (about one second) in an isolated session of its configured backend. No arguments, live capture, playback, provider switching, settings changes or transcript/history/translation/output publication. Inspect the terminal receipt. Success means nonempty final synthetic recognition only, not accuracy, other-language quality, microphone/loopback capture or display. The existing configured connection may incur ordinary recognition cost.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "probe_translation",
    "description": "Run one synthetic translation through the current attached usable backend. No arguments, live conversation, custom prompt, history, capture activation, provider provisioning or output publication. Inspect the terminal receipt; success tests translation only, not output delivery or display. Existing runtime routing policy remains in effect; this does not prove primary-model recovery.",
    "strict": true,
    "parameters": {
      "type": "object",
      "properties": {},
      "required": [],
      "additionalProperties": false
    }
  }
];
