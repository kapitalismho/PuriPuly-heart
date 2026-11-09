export const appTools = [
  {
    "type": "function",
    "name": "read_app",
    "description": "Observe current app state without UI or runtime effects. topic: a context module or overview; diagnostics with optional diagnostic_id, operation_id or after; gpu, models, auth or peer_consent status; provider_setup with provider; operation with operation_id for a retained receipt.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "topic": {
          "type": "string",
          "enum": [
            "accounts",
            "audio",
            "auth",
            "general",
            "gpu",
            "languages",
            "models",
            "output",
            "overview",
            "peer_consent",
            "recognition",
            "translation",
            "diagnostics",
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
            "gemini_transcribe",
            "google",
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
    "name": "execute_app",
    "description": "Run an immediate operation or stage existing UI controls. Parameters: capture.set channel and enabled; translation.set and overlay.set enabled; recognition.probe channel; ui.provider_setup provider; ui.settings section, optional provider; confirmation.withdraw card_id. Probes are active scoped checks, not proof of live capture or display; audio.diagnose measures local capture/VAD and never changes settings.",
    "strict": false,
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "capture.set",
            "translation.set",
            "overlay.set",
            "recognition.probe",
            "translation.probe",
            "audio.diagnose",
            "audio.retry",
            "osc.discovery.retry",
            "ui.accounts",
            "models.retry",
            "ui.provider_setup",
            "ui.settings",
            "confirmation.withdraw"
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
        },
        "backend": {
          "type": "string",
          "enum": [
            "gpu"
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
        "card_id": {
          "type": "string",
          "minLength": 1,
          "maxLength": 64
        }
      },
      "required": [
        "action"
      ],
      "additionalProperties": false
    }
  },
  {
    "type": "function",
    "name": "propose_app",
    "description": "Queue a user approval card: settings with sparse changes (only changed fields, never null placeholders), secrets.delete name, auth.logout provider, overlay.position.reset, or report summary, which offers a send button.",
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
            "secrets.delete"
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
