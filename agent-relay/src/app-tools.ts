export const appTools = [
  {
    "type": "function",
    "name": "read_app",
    "description": "Observe current app state without UI or runtime effects. Arguments: accounts; audio; general; languages; output; overview; recognition; translation; diagnostics(diagnostic_id?, operation_id?, after?); gpu; models; auth; peer_consent; provider_setup(provider); operation(operation_id). Retained receipt; resolve pending/uncertain work by its identity, not resubmission. Supply only the selected action's fields.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "topic": {
          "type": "string",
          "enum": [
            "accounts",
            "audio",
            "general",
            "languages",
            "output",
            "overview",
            "recognition",
            "translation",
            "diagnostics",
            "gpu",
            "models",
            "auth",
            "peer_consent",
            "provider_setup",
            "operation"
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
        },
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
        "topic"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "set_app",
    "description": "Set an existing runtime switch; no saved settings change. Arguments: capture.self(enabled); capture.peer(enabled); translation(enabled); overlay(enabled). Supply only the selected action's fields.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "target": {
          "type": "string",
          "enum": [
            "capture.self",
            "capture.peer",
            "translation",
            "overlay"
          ]
        },
        "enabled": {
          "type": "boolean"
        }
      },
      "required": [
        "target",
        "enabled"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "probe_app",
    "description": "Run an isolated scoped check, not live pipeline repair or output delivery. Arguments: recognition.self; recognition.peer; translation.self; translation.peer; microphone. Local capture/VAD only; no settings change. Supply only the selected action's fields.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "target": {
          "type": "string",
          "enum": [
            "recognition.self",
            "recognition.peer",
            "translation.self",
            "translation.peer",
            "microphone"
          ]
        }
      },
      "required": [
        "target"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "retry_app",
    "description": "Retry existing application recovery through its resource owner. Arguments: audio; osc_discovery; gpu_activation. Supply only the selected action's fields.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "target": {
          "type": "string",
          "enum": [
            "audio",
            "osc_discovery",
            "gpu_activation"
          ]
        }
      },
      "required": [
        "target"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "open_app",
    "description": "Stage existing app controls; staging is not display, click, verification or readiness. Arguments: accounts; provider_setup(provider); settings(section, provider?). Supply only the selected action's fields.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "view": {
          "type": "string",
          "enum": [
            "accounts",
            "provider_setup",
            "settings"
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
            "gemini",
            "openai",
            "openrouter",
            "soniox"
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
        }
      },
      "required": [
        "view"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "manage_proposal",
    "description": "Create or withdraw a user card, never approve or send it. Arguments: settings(changes). Sparse changes only; absent fields stay unchanged, never null placeholders; report(summary). Offers a send button; upload requires the user's click; auth.logout(provider); overlay.position.reset; secrets.delete(name); withdraw(card_id). Only the matching pending card; accepted application work is not cancelled. Supply only the selected action's fields.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "settings",
            "report",
            "auth.logout",
            "overlay.position.reset",
            "secrets.delete",
            "withdraw"
          ]
        },
        "changes": {
          "type": "object",
          "properties": {
            "audio.input_device": {
              "type": "string"
            },
            "audio.input_host_api": {
              "type": "string"
            },
            "audio.output_device": {
              "type": "string"
            },
            "chatbox.activation_notice.enabled": {
              "type": "boolean"
            },
            "chatbox.source.enabled": {
              "type": "boolean"
            },
            "clipboard.enabled": {
              "type": "boolean"
            },
            "languages": {
              "type": "object",
              "properties": {
                "source": {
                  "$ref": "#/$defs/language"
                },
                "target": {
                  "$ref": "#/$defs/language"
                },
                "secondary_target": {
                  "$ref": "#/$defs/optional_language"
                },
                "peer_source": {
                  "$ref": "#/$defs/optional_language"
                },
                "peer_target": {
                  "$ref": "#/$defs/optional_language"
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
                "en",
                "ko",
                "zh-CN",
                "ja",
                "ru"
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
              "type": "boolean"
            },
            "overlay.swap_caption_languages": {
              "type": "boolean"
            },
            "overlay.target": {
              "type": "string",
              "enum": [
                "steamvr",
                "desktop"
              ]
            },
            "overlay.translation.enabled": {
              "type": "boolean"
            },
            "peer_expected_languages": {
              "type": "array",
              "items": {
                "$ref": "#/$defs/language"
              }
            },
            "peer_stt.provider": {
              "$ref": "#/$defs/stt_provider"
            },
            "peer_vad.hangover_ms": {
              "type": "integer",
              "minimum": 0
            },
            "peer_vad.pre_roll_ms": {
              "type": "integer",
              "minimum": 0
            },
            "peer_vad.speech_threshold": {
              "type": "number",
              "minimum": 0.1,
              "maximum": 1
            },
            "self_vad.speech_threshold": {
              "type": "number",
              "minimum": 0.1,
              "maximum": 1
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
              "type": "string"
            },
            "stt.provider": {
              "$ref": "#/$defs/stt_provider"
            },
            "stt.soniox.enable_speaker_diarization": {
              "type": "boolean"
            },
            "telemetry.enabled": {
              "type": "boolean"
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
              "type": "string"
            },
            "translation.http_extension_id": {
              "type": "string"
            },
            "translation.model": {
              "$ref": "#/$defs/translation_model"
            },
            "translation.previous_llm_model": {
              "$ref": "#/$defs/translation_model"
            },
            "translation.qwen.region": {
              "type": "string",
              "enum": [
                "beijing",
                "singapore"
              ]
            },
            "vrc_mic_intercept.enabled": {
              "type": "boolean"
            }
          },
          "required": [],
          "additionalProperties": false
        },
        "summary": {
          "type": "string",
          "minLength": 1,
          "maxLength": 2000
        },
        "provider": {
          "type": "string",
          "enum": [
            "qq",
            "discord",
            "openrouter",
            "chatgpt"
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
        },
        "card_id": {
          "type": "string",
          "minLength": 1,
          "maxLength": 64
        }
      },
      "required": [
        "action"
      ],
      "additionalProperties": false,
      "$defs": {
        "language": {
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
        "optional_language": {
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
        "stt_provider": {
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
        "translation_model": {
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
