export const actionTools = [
  {
    "type": "function",
    "name": "get_app_context",
    "description": "Read missing or changed app evidence; the initial overview already includes settings choices, audio routes, readiness and coarse cached accounts. An explicit accounts read also stages user-click account actions; initial facts alone do not show controls. No identities, secrets or live balance refresh. Diagnostics accepts observed diagnostic_id/operation_id or next_after, returning at most 10 records; missing/evicted records do not prove recovery. Other topics require null filters. No raw logs or runtime changes.",
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
            "output",
            "overview",
            "settings",
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
                    "en",
                    "ko",
                    "zh-CN",
                    "ja",
                    "ru"
                  ]
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
            "translation.model": {
              "anyOf": [
                {
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
            "languages",
            "locale",
            "overlay.background_alpha",
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
            "stt.provider",
            "translation.connection",
            "translation.model"
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
    "name": "withdraw_confirmation",
    "description": "Withdraw the exact current pending settings card identified by app_context.pending_confirmation.id and its remaining unapproved cards when the user cancels that proposal. A question about the card or an unrelated message does not withdraw it. Stale or unknown IDs are rejected. This never approves, applies or undoes settings, and creates no app-notice continuation.",
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
            "gemini_transcribe",
            "google",
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
            "gemini_transcribe",
            "google",
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
