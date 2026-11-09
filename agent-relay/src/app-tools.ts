export const appTools = [
  {
    "type": "function",
    "name": "read_app",
    "description": "Observe app context, allowlisted queries or provider setup metadata without UI or runtime effects. Independent reads may share a batch; results are not an atomic snapshot. Use observed diagnostic IDs/cursors for retained details; not_retained is not an operation failure.",
    "strict": false,
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
                    "accounts.status",
                    "app.status",
                    "audio.devices",
                    "auth.status",
                    "capture.channels",
                    "consent.peer_translation",
                    "diagnostics.health",
                    "gpu.status",
                    "models.status",
                    "osc.status",
                    "overlay.status",
                    "providers.status",
                    "secrets.presence",
                    "settings.choices",
                    "settings.current"
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
                    "diagnostics.history"
                  ]
                },
                "operation_id": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 128
                },
                "diagnostic_id": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 128
                },
                "after": {
                  "type": "integer",
                  "minimum": 0
                },
                "limit": {
                  "type": "integer",
                  "minimum": 1,
                  "maximum": 100
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
                    "context"
                  ]
                },
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
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 64
                },
                "operation_id": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 64
                },
                "after": {
                  "type": "integer",
                  "minimum": 0
                }
              },
              "required": [
                "action",
                "topic"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "provider.setup"
                  ]
                },
                "provider": {
                  "type": "string",
                  "enum": [
                    "alibaba_beijing",
                    "alibaba_singapore",
                    "deepgram",
                    "deepseek",
                    "elevenlabs_scribe",
                    "gemini_transcribe",
                    "google",
                    "openai",
                    "openrouter",
                    "soniox"
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
                    "operation"
                  ]
                },
                "operation_id": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 64
                }
              },
              "required": [
                "action",
                "operation_id"
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
    "name": "execute_app",
    "description": "Run listed immediate operations or explicitly stage existing UI controls. ui.provider_setup returns metadata and stages links/secure entry; ui.accounts stages account buttons. Staged is not displayed, clicked, verified or ready. Only users enter credentials and authorize accounts. Probes are active, scoped synthetic checks, not proof of live capture, quality or display. audio.diagnose exercises local capture/VAD, not recognition, and never changes settings. OSC target application does not prove remote receipt. Inspect terminal receipts; do not replay uncertain work. confirmation.withdraw removes only the exact pending proposal, never saved changes.",
    "strict": false,
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
                    "capture.set"
                  ]
                },
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
                "action",
                "channel",
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
                    "translation.set",
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
                    "recognition.probe"
                  ]
                },
                "channel": {
                  "type": "string",
                  "enum": [
                    "self",
                    "peer"
                  ]
                }
              },
              "required": [
                "action",
                "channel"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "audio.diagnose",
                    "osc.discovery.retry",
                    "translation.probe",
                    "audio.retry",
                    "ui.accounts"
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
                    "models.retry"
                  ]
                },
                "backend": {
                  "type": "string",
                  "enum": [
                    "gpu"
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
                    "ui.provider_setup"
                  ]
                },
                "provider": {
                  "type": "string",
                  "enum": [
                    "alibaba_beijing",
                    "alibaba_singapore",
                    "deepgram",
                    "deepseek",
                    "elevenlabs_scribe",
                    "gemini_transcribe",
                    "google",
                    "openai",
                    "openrouter",
                    "soniox"
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
                    "ui.settings"
                  ]
                },
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
                  "type": "string",
                  "enum": [
                    "alibaba_beijing",
                    "alibaba_singapore",
                    "deepgram",
                    "deepseek",
                    "elevenlabs_scribe",
                    "gemini_transcribe",
                    "google",
                    "openai",
                    "openrouter",
                    "soniox"
                  ]
                }
              },
              "required": [
                "action",
                "section"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "confirmation.withdraw"
                  ]
                },
                "card_id": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 64
                }
              },
              "required": [
                "action",
                "card_id"
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
    "name": "propose_app",
    "description": "Propose sparse typed settings, key deletion, logout, position reset or a support report. Settings require revision-bound user approval; supply only changed fields, never null placeholders. Submit translation model and connection together. Coupled audio routes retain validation. The app queues cards with at most two blocks; explain and end the turn without resubmitting. Key deletion is local, not provider revocation. Reports upload only after the user's explicit click. Approval, required user action, pending work or blocking failure stops later effects.",
    "strict": false,
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
                    "settings"
                  ]
                },
                "changes": {
                  "type": "object",
                  "properties": {
                    "audio.input_device": {
                      "$ref": "#/$defs/value1"
                    },
                    "audio.input_host_api": {
                      "$ref": "#/$defs/value1"
                    },
                    "audio.output_device": {
                      "$ref": "#/$defs/value1"
                    },
                    "chatbox.activation_notice.enabled": {
                      "$ref": "#/$defs/value2"
                    },
                    "chatbox.source.enabled": {
                      "$ref": "#/$defs/value2"
                    },
                    "clipboard.enabled": {
                      "$ref": "#/$defs/value2"
                    },
                    "languages": {
                      "type": "object",
                      "properties": {
                        "source": {
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
                        "target": {
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
                        "secondary_target": {
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
                        "peer_source": {
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
                        "peer_target": {
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
                        "peer_source_mode": {
                          "type": "string",
                          "enum": [
                            "manual",
                            "auto"
                          ]
                        }
                      },
                      "required": [],
                      "additionalProperties": false
                    },
                    "locale": {
                      "type": "string",
                      "enum": [
                        "ko",
                        "en",
                        "ja"
                      ]
                    },
                    "osc.connection": {
                      "type": "object",
                      "properties": {
                        "mode": {
                          "type": "string",
                          "enum": [
                            "automatic",
                            "manual",
                            "off"
                          ]
                        },
                        "send_port": {
                          "type": "integer",
                          "minimum": 1,
                          "maximum": 65535
                        },
                        "receive_port": {
                          "type": "integer",
                          "minimum": 1,
                          "maximum": 65535
                        }
                      },
                      "required": [],
                      "additionalProperties": false
                    },
                    "overlay.background_alpha": {
                      "type": "number",
                      "minimum": 0,
                      "maximum": 1
                    },
                    "overlay.desktop_size": {
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
                    "overlay.peer_original.enabled": {
                      "$ref": "#/$defs/value2"
                    },
                    "overlay.swap_caption_languages": {
                      "$ref": "#/$defs/value2"
                    },
                    "overlay.target": {
                      "type": "string",
                      "enum": [
                        "steamvr",
                        "desktop"
                      ]
                    },
                    "overlay.translation.enabled": {
                      "$ref": "#/$defs/value2"
                    },
                    "peer_expected_languages": {
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
                    "peer_stt.provider": {
                      "$ref": "#/$defs/value3"
                    },
                    "peer_vad.hangover_ms": {
                      "$ref": "#/$defs/value4"
                    },
                    "peer_vad.pre_roll_ms": {
                      "$ref": "#/$defs/value4"
                    },
                    "peer_vad.speech_threshold": {
                      "$ref": "#/$defs/value5"
                    },
                    "self_vad.speech_threshold": {
                      "$ref": "#/$defs/value5"
                    },
                    "stt.cloud_free_tier_providers": {
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
                    "stt.gpu_device_id": {
                      "$ref": "#/$defs/value1"
                    },
                    "stt.provider": {
                      "$ref": "#/$defs/value3"
                    },
                    "stt.soniox.enable_speaker_diarization": {
                      "$ref": "#/$defs/value2"
                    },
                    "telemetry.enabled": {
                      "$ref": "#/$defs/value2"
                    },
                    "translation.connection": {
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
                    "translation.connection_history": {
                      "type": "object",
                      "properties": {
                        "index_translate_35b": {
                          "type": "string",
                          "enum": [
                            "public_api"
                          ]
                        },
                        "gemma4_26b_31b": {
                          "type": "string",
                          "enum": [
                            "managed",
                            "openrouter"
                          ]
                        },
                        "deepseek_v4_flash": {
                          "type": "string",
                          "enum": [
                            "openrouter"
                          ]
                        },
                        "deepseek_v4_flash_41": {
                          "type": "string",
                          "enum": [
                            "managed",
                            "managed_china",
                            "openrouter",
                            "official_byok"
                          ]
                        },
                        "gemini_flash": {
                          "type": "string",
                          "enum": [
                            "official_byok",
                            "openrouter"
                          ]
                        },
                        "gpt_6_luna": {
                          "type": "string",
                          "enum": [
                            "chatgpt",
                            "openrouter",
                            "official_byok"
                          ]
                        },
                        "qwen38_flash": {
                          "type": "string",
                          "enum": [
                            "official_byok"
                          ]
                        },
                        "managed_gemma": {
                          "type": "string",
                          "enum": [
                            "cpu",
                            "gpu"
                          ]
                        },
                        "managed_gemma_26b": {
                          "type": "string",
                          "enum": [
                            "gpu"
                          ]
                        },
                        "local_llm": {
                          "type": "string",
                          "enum": [
                            "ollama"
                          ]
                        },
                        "custom_http": {
                          "type": "string",
                          "enum": [
                            "custom_http"
                          ]
                        }
                      },
                      "required": [],
                      "additionalProperties": false
                    },
                    "translation.gpu_device_id": {
                      "$ref": "#/$defs/value1"
                    },
                    "translation.http_extension_id": {
                      "$ref": "#/$defs/value1"
                    },
                    "translation.model": {
                      "$ref": "#/$defs/value6"
                    },
                    "translation.previous_llm_model": {
                      "$ref": "#/$defs/value6"
                    },
                    "translation.qwen.region": {
                      "type": "string",
                      "enum": [
                        "beijing",
                        "singapore"
                      ]
                    },
                    "vrc_mic_intercept.enabled": {
                      "$ref": "#/$defs/value2"
                    }
                  },
                  "required": [],
                  "additionalProperties": false
                }
              },
              "required": [
                "action",
                "changes"
              ],
              "additionalProperties": false
            },
            {
              "type": "object",
              "properties": {
                "action": {
                  "type": "string",
                  "enum": [
                    "report"
                  ]
                },
                "summary": {
                  "type": "string",
                  "minLength": 1,
                  "maxLength": 2000
                }
              },
              "required": [
                "action",
                "summary"
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
                    "secrets.delete"
                  ]
                },
                "name": {
                  "type": "string",
                  "enum": [
                    "google_api_key",
                    "openrouter_api_key",
                    "openai_api_key",
                    "deepseek_api_key",
                    "deepgram_api_key",
                    "gemini_transcribe_api_key",
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
      "additionalProperties": false,
      "$defs": {
        "value1": {
          "type": "string"
        },
        "value2": {
          "type": "boolean"
        },
        "value3": {
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
        "value4": {
          "type": "integer",
          "minimum": 0
        },
        "value5": {
          "type": "number",
          "minimum": 0.1,
          "maximum": 1
        },
        "value6": {
          "type": "string",
          "enum": [
            "index_translate_35b",
            "gemma4_26b_31b",
            "deepseek_v4_flash",
            "deepseek_v4_flash_41",
            "gemini_flash",
            "gpt_6_luna",
            "qwen38_flash",
            "managed_gemma",
            "managed_gemma_26b",
            "local_llm",
            "custom_http"
          ]
        }
      }
    }
  }
];
