use std::path::{Path, PathBuf};

use super::types::contains_cjk;

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
#[cfg(windows)]
use windows::core::{Interface, PCWSTR};
#[cfg(windows)]
use windows::Win32::Globalization::GetUserDefaultUILanguage;
#[cfg(windows)]
use windows::Win32::Graphics::DirectWrite::{
    DWriteCreateFactory, IDWriteFactory, IDWriteFactory3, IDWriteFactory5, IDWriteFontCollection,
    DWRITE_FACTORY_TYPE_SHARED,
};

pub const BUNDLED_NOTO_CJK_FILE_NAME: &str = "NotoSansCJK-Medium.ttc";
const DIRECTWRITE_SYSTEM_FALLBACK: &str = "DirectWrite system fallback";

const GENERAL_SYSTEM_FALLBACKS: &[&str] = &["Noto Sans", "Segoe UI", DIRECTWRITE_SYSTEM_FALLBACK];
const KOREAN_SYSTEM_FALLBACKS: &[&str] =
    &["Malgun Gothic", "Segoe UI", DIRECTWRITE_SYSTEM_FALLBACK];
const JAPANESE_SYSTEM_FALLBACKS: &[&str] = &[
    "Yu Gothic UI",
    "Meiryo UI",
    "Segoe UI",
    DIRECTWRITE_SYSTEM_FALLBACK,
];
const SIMPLIFIED_CHINESE_SYSTEM_FALLBACKS: &[&str] = &[
    "Microsoft YaHei UI",
    "Segoe UI",
    DIRECTWRITE_SYSTEM_FALLBACK,
];
const TRADITIONAL_CHINESE_SYSTEM_FALLBACKS: &[&str] = &[
    "Microsoft JhengHei UI",
    "Segoe UI",
    DIRECTWRITE_SYSTEM_FALLBACK,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontLanguageBucket {
    General,
    CjkKo,
    CjkJa,
    CjkZhHans,
    CjkZhHant,
}

impl FontLanguageBucket {
    pub fn for_text(language: Option<&str>, text: &str) -> Self {
        Self::for_text_with_ui_language(language, text, None)
    }

    pub fn for_text_with_ui_language(
        language: Option<&str>,
        text: &str,
        ui_language_hint: Option<&str>,
    ) -> Self {
        match normalize_language_bucket(language) {
            LanguageBucketNormalization::Known(bucket) => bucket,
            LanguageBucketNormalization::Unknown => {
                if contains_cjk(text) {
                    ui_language_bucket_hint(ui_language_hint).unwrap_or_else(unknown_cjk_bucket)
                } else {
                    Self::General
                }
            }
        }
    }

    pub fn directwrite_locale(self, language: Option<&str>) -> String {
        match self {
            Self::General => normalize_explicit_locale(language).unwrap_or_else(|| "en-US".into()),
            Self::CjkKo => "ko-KR".into(),
            Self::CjkJa => "ja-JP".into(),
            Self::CjkZhHans => "zh-CN".into(),
            Self::CjkZhHant => "zh-TW".into(),
        }
    }

    pub fn system_fallback_families(self) -> &'static [&'static str] {
        match self {
            Self::General => GENERAL_SYSTEM_FALLBACKS,
            Self::CjkKo => KOREAN_SYSTEM_FALLBACKS,
            Self::CjkJa => JAPANESE_SYSTEM_FALLBACKS,
            Self::CjkZhHans => SIMPLIFIED_CHINESE_SYSTEM_FALLBACKS,
            Self::CjkZhHant => TRADITIONAL_CHINESE_SYSTEM_FALLBACKS,
        }
    }

    fn bundled_face(self) -> Option<BundledFaceId> {
        match self {
            Self::General => None,
            Self::CjkKo => Some(BundledFaceId::NotoCjkKrMedium),
            Self::CjkJa => Some(BundledFaceId::NotoCjkJpMedium),
            Self::CjkZhHans => Some(BundledFaceId::NotoCjkScMedium),
            Self::CjkZhHant => Some(BundledFaceId::NotoCjkTcMedium),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BundledFaceId {
    NotoCjkKrMedium,
    NotoCjkJpMedium,
    NotoCjkScMedium,
    NotoCjkTcMedium,
}

impl BundledFaceId {
    pub fn family_name(self) -> &'static str {
        match self {
            Self::NotoCjkKrMedium => "Noto Sans CJK KR",
            Self::NotoCjkJpMedium => "Noto Sans CJK JP",
            Self::NotoCjkScMedium => "Noto Sans CJK SC",
            Self::NotoCjkTcMedium => "Noto Sans CJK TC",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontSource {
    BundledNotoCjkMedium,
    SystemFont,
    SystemFallbackSentinel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontWeight {
    Regular,
    Medium,
    SemiBold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextFamilyKey {
    NotoSans,
    SegoeUi,
    MalgunGothic,
    YuGothicUi,
    MeiryoUi,
    MicrosoftYaHeiUi,
    MicrosoftJhengHeiUi,
    NotoSansCjkKr,
    NotoSansCjkJp,
    NotoSansCjkSc,
    NotoSansCjkTc,
    Other(u64),
}

impl TextFamilyKey {
    pub fn from_family_name(family_name: &str) -> Self {
        match family_name {
            "Noto Sans" => Self::NotoSans,
            "Segoe UI" => Self::SegoeUi,
            "Malgun Gothic" => Self::MalgunGothic,
            "Yu Gothic UI" => Self::YuGothicUi,
            "Meiryo UI" => Self::MeiryoUi,
            "Microsoft YaHei UI" => Self::MicrosoftYaHeiUi,
            "Microsoft JhengHei UI" => Self::MicrosoftJhengHeiUi,
            "Noto Sans CJK KR" => Self::NotoSansCjkKr,
            "Noto Sans CJK JP" => Self::NotoSansCjkJp,
            "Noto Sans CJK SC" => Self::NotoSansCjkSc,
            "Noto Sans CJK TC" => Self::NotoSansCjkTc,
            other => Self::Other(stable_compact_hash(other)),
        }
    }

    pub fn family_name(self) -> Option<&'static str> {
        match self {
            Self::NotoSans => Some("Noto Sans"),
            Self::SegoeUi => Some("Segoe UI"),
            Self::MalgunGothic => Some("Malgun Gothic"),
            Self::YuGothicUi => Some("Yu Gothic UI"),
            Self::MeiryoUi => Some("Meiryo UI"),
            Self::MicrosoftYaHeiUi => Some("Microsoft YaHei UI"),
            Self::MicrosoftJhengHeiUi => Some("Microsoft JhengHei UI"),
            Self::NotoSansCjkKr => Some("Noto Sans CJK KR"),
            Self::NotoSansCjkJp => Some("Noto Sans CJK JP"),
            Self::NotoSansCjkSc => Some("Noto Sans CJK SC"),
            Self::NotoSansCjkTc => Some("Noto Sans CJK TC"),
            Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextLocaleKey {
    EnUs,
    KoKr,
    JaJp,
    ZhCn,
    ZhTw,
    Other(u64),
}

impl TextLocaleKey {
    pub fn from_locale(locale: &str) -> Self {
        match locale {
            "en-US" => Self::EnUs,
            "ko-KR" => Self::KoKr,
            "ja-JP" => Self::JaJp,
            "zh-CN" => Self::ZhCn,
            "zh-TW" => Self::ZhTw,
            other => Self::Other(stable_compact_hash(other)),
        }
    }

    pub fn locale_name(self) -> Option<&'static str> {
        match self {
            Self::EnUs => Some("en-US"),
            Self::KoKr => Some("ko-KR"),
            Self::JaJp => Some("ja-JP"),
            Self::ZhCn => Some("zh-CN"),
            Self::ZhTw => Some("zh-TW"),
            Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextStyleKey {
    pub bucket: FontLanguageBucket,
    pub source: FontSource,
    pub bundled_face: Option<BundledFaceId>,
    pub family: TextFamilyKey,
    pub weight: FontWeight,
    pub locale: TextLocaleKey,
}

impl TextStyleKey {
    pub fn from_parts(
        bucket: FontLanguageBucket,
        source: FontSource,
        bundled_face: Option<BundledFaceId>,
        family_name: &str,
        weight: FontWeight,
        locale: &str,
    ) -> Self {
        Self {
            bucket,
            source,
            bundled_face,
            family: TextFamilyKey::from_family_name(family_name),
            weight,
            locale: TextLocaleKey::from_locale(locale),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontFallbackReason {
    BundledFontUnavailable,
    DirectWriteStyleResolutionFailure,
    CatastrophicDirectWriteLayoutFailure,
}

impl FontFallbackReason {
    pub fn log_label(self) -> &'static str {
        match self {
            Self::BundledFontUnavailable => "bundled_font_unavailable",
            Self::DirectWriteStyleResolutionFailure => "directwrite_style_resolution_failure",
            Self::CatastrophicDirectWriteLayoutFailure => "catastrophic_directwrite_layout_failure",
        }
    }

    pub fn uses_heuristic_layout_fallback(self) -> bool {
        matches!(self, Self::CatastrophicDirectWriteLayoutFailure)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFontStyle {
    pub bucket: FontLanguageBucket,
    pub source: FontSource,
    pub bundled_face: Option<BundledFaceId>,
    pub family_name: &'static str,
    pub weight: FontWeight,
    pub locale: String,
    system_fallback_families: &'static [&'static str],
    pub fallback_reason: Option<FontFallbackReason>,
}

impl ResolvedFontStyle {
    pub fn style_key(&self) -> TextStyleKey {
        TextStyleKey::from_parts(
            self.bucket,
            self.source,
            self.bundled_face,
            self.family_name,
            self.weight,
            &self.locale,
        )
    }

    pub fn system_fallback_families(&self) -> &'static [&'static str] {
        self.system_fallback_families
    }

    pub fn fallback_chain(&self) -> Vec<&'static str> {
        if self.source == FontSource::BundledNotoCjkMedium {
            let mut chain = Vec::with_capacity(self.system_fallback_families.len() + 1);
            chain.push(self.family_name);
            chain.extend_from_slice(self.system_fallback_families);
            chain
        } else {
            self.system_fallback_families.to_vec()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FontResolver {
    bundled_font_available: bool,
    bundled_font_failure: Option<String>,
    ui_language_hint: Option<String>,
}

impl Default for FontResolver {
    fn default() -> Self {
        Self::with_bundle_unavailable("bundled font availability has not been initialized")
    }
}

impl FontResolver {
    pub fn with_bundle_available() -> Self {
        Self {
            bundled_font_available: true,
            bundled_font_failure: None,
            ui_language_hint: None,
        }
    }

    pub fn with_bundle_unavailable(reason: impl Into<String>) -> Self {
        Self {
            bundled_font_available: false,
            bundled_font_failure: Some(reason.into()),
            ui_language_hint: None,
        }
    }

    pub fn with_ui_language_hint(mut self, language: impl Into<String>) -> Self {
        let language = language.into();
        self.ui_language_hint = clean_language_tag(&language);
        self
    }

    pub(crate) fn with_optional_ui_language_hint(mut self, language: Option<String>) -> Self {
        self.ui_language_hint = language.and_then(|language| clean_language_tag(&language));
        self
    }

    pub fn bundled_font_failure(&self) -> Option<&str> {
        self.bundled_font_failure.as_deref()
    }

    pub fn resolve(&self, language: Option<&str>, text: &str) -> ResolvedFontStyle {
        let bucket = FontLanguageBucket::for_text_with_ui_language(
            language,
            text,
            self.ui_language_hint.as_deref(),
        );
        let locale = bucket.directwrite_locale(language);
        let system_fallback_families = bucket.system_fallback_families();

        if let Some(bundled_face) = bucket.bundled_face() {
            if self.bundled_font_available {
                return ResolvedFontStyle {
                    bucket,
                    source: FontSource::BundledNotoCjkMedium,
                    bundled_face: Some(bundled_face),
                    family_name: bundled_face.family_name(),
                    weight: FontWeight::Medium,
                    locale,
                    system_fallback_families,
                    fallback_reason: None,
                };
            }

            return ResolvedFontStyle {
                bucket,
                source: FontSource::SystemFont,
                bundled_face: None,
                family_name: first_real_family(system_fallback_families),
                weight: FontWeight::Regular,
                locale,
                system_fallback_families,
                fallback_reason: Some(FontFallbackReason::BundledFontUnavailable),
            };
        }

        ResolvedFontStyle {
            bucket,
            source: FontSource::SystemFont,
            bundled_face: None,
            family_name: first_real_family(system_fallback_families),
            weight: FontWeight::Regular,
            locale,
            system_fallback_families,
            fallback_reason: None,
        }
    }

    pub fn style_resolution_failure_fallback(bucket: FontLanguageBucket) -> ResolvedFontStyle {
        Self::style_resolution_failure_fallback_for_bucket_locale(
            bucket,
            bucket.directwrite_locale(None),
        )
    }

    pub fn style_resolution_failure_fallback_for_style(
        style: &ResolvedFontStyle,
    ) -> ResolvedFontStyle {
        Self::style_resolution_failure_fallback_for_bucket_locale(
            style.bucket,
            style.locale.clone(),
        )
    }

    pub(crate) fn style_resolution_failure_fallback_for_bucket_locale(
        bucket: FontLanguageBucket,
        locale: String,
    ) -> ResolvedFontStyle {
        ResolvedFontStyle {
            bucket,
            source: FontSource::SystemFallbackSentinel,
            bundled_face: None,
            family_name: "Segoe UI",
            weight: FontWeight::Regular,
            locale,
            system_fallback_families: &["Segoe UI", DIRECTWRITE_SYSTEM_FALLBACK],
            fallback_reason: Some(FontFallbackReason::DirectWriteStyleResolutionFailure),
        }
    }
}

pub fn bundled_font_path_from_exe_dir(exe_dir: &Path) -> PathBuf {
    exe_dir
        .join("puripuly_heart")
        .join("data")
        .join("fonts")
        .join(BUNDLED_NOTO_CJK_FILE_NAME)
}

pub fn runtime_bundled_font_path() -> Result<PathBuf, std::io::Error> {
    let current_exe = std::env::current_exe()?;
    let exe_dir = current_exe.parent().unwrap_or_else(|| Path::new("."));
    Ok(bundled_font_path_from_exe_dir(exe_dir))
}

pub fn system_ui_language_hint() -> Option<String> {
    let language = system_ui_language_hint_impl()?;
    ui_language_bucket_hint(Some(&language)).map(|_| language)
}

#[cfg(windows)]
fn system_ui_language_hint_impl() -> Option<String> {
    ui_language_hint_from_windows_langid(unsafe { GetUserDefaultUILanguage() })
}

#[cfg(not(windows))]
fn system_ui_language_hint_impl() -> Option<String> {
    None
}

fn ui_language_hint_from_windows_langid(langid: u16) -> Option<String> {
    const LANG_JAPANESE: u16 = 0x11;
    const LANG_KOREAN: u16 = 0x12;
    const LANG_CHINESE: u16 = 0x04;
    const SUBLANG_CHINESE_TRADITIONAL: u16 = 0x01;
    const SUBLANG_CHINESE_SIMPLIFIED: u16 = 0x02;
    const SUBLANG_CHINESE_HONGKONG: u16 = 0x03;
    const SUBLANG_CHINESE_SINGAPORE: u16 = 0x04;
    const SUBLANG_CHINESE_MACAU: u16 = 0x05;

    let primary_language = langid & 0x03ff;
    let sublanguage = langid >> 10;
    let language = match primary_language {
        LANG_KOREAN => "ko-KR",
        LANG_JAPANESE => "ja-JP",
        LANG_CHINESE => match sublanguage {
            SUBLANG_CHINESE_TRADITIONAL => "zh-TW",
            SUBLANG_CHINESE_SIMPLIFIED | SUBLANG_CHINESE_SINGAPORE => "zh-CN",
            SUBLANG_CHINESE_HONGKONG | SUBLANG_CHINESE_MACAU => "zh-HK",
            _ => return None,
        },
        _ => return None,
    };
    Some(language.to_string())
}

#[cfg(windows)]
#[derive(Debug, Clone)]
pub struct WindowsBundledFontCollection {
    path: PathBuf,
    collection: IDWriteFontCollection,
}

#[cfg(windows)]
impl WindowsBundledFontCollection {
    pub fn load_from_path(path: &Path) -> Result<Self, String> {
        let factory: IDWriteFactory = unsafe { DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED) }
            .map_err(|error| format!("create DirectWrite factory: {error}"))?;
        Self::load_with_factory(&factory, path)
    }

    pub(crate) fn load_with_factory(factory: &IDWriteFactory, path: &Path) -> Result<Self, String> {
        let absolute_path = path
            .canonicalize()
            .map_err(|error| format!("canonicalize bundled font path {path:?}: {error}"))?;
        let factory5: IDWriteFactory5 = factory
            .cast()
            .map_err(|error| format!("DirectWrite factory5 unavailable: {error}"))?;
        let factory3: IDWriteFactory3 = factory
            .cast()
            .map_err(|error| format!("DirectWrite factory3 unavailable: {error}"))?;
        let path_wide = utf16_path_null(&absolute_path);
        let font_file = unsafe {
            factory
                .CreateFontFileReference(PCWSTR::from_raw(path_wide.as_ptr()), None)
                .map_err(|error| format!("create bundled font file reference: {error}"))?
        };
        let builder = unsafe {
            factory5
                .CreateFontSetBuilder()
                .map_err(|error| format!("create bundled font set builder: {error}"))?
        };
        unsafe {
            builder
                .AddFontFile(&font_file)
                .map_err(|error| format!("add bundled TTC to font set: {error}"))?;
        }
        let font_set = unsafe {
            builder
                .CreateFontSet()
                .map_err(|error| format!("create bundled font set: {error}"))?
        };
        let collection1 = unsafe {
            factory3
                .CreateFontCollectionFromFontSet(&font_set)
                .map_err(|error| format!("create bundled font collection: {error}"))?
        };
        let collection = collection1
            .cast()
            .map_err(|error| format!("upcast bundled font collection: {error}"))?;

        Ok(Self {
            path: absolute_path,
            collection,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn collection(&self) -> &IDWriteFontCollection {
        &self.collection
    }
}

#[cfg(windows)]
fn utf16_path_null(path: &Path) -> Vec<u16> {
    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn first_real_family(families: &'static [&'static str]) -> &'static str {
    families
        .iter()
        .copied()
        .find(|family| *family != DIRECTWRITE_SYSTEM_FALLBACK)
        .unwrap_or("Segoe UI")
}

fn stable_compact_hash(value: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    value.bytes().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
    })
}

fn unknown_cjk_bucket() -> FontLanguageBucket {
    // Compatibility default: old CJK handling was KR-first.
    FontLanguageBucket::CjkKo
}

fn ui_language_bucket_hint(ui_language_hint: Option<&str>) -> Option<FontLanguageBucket> {
    match normalize_language_bucket(ui_language_hint) {
        LanguageBucketNormalization::Known(FontLanguageBucket::General)
        | LanguageBucketNormalization::Unknown => None,
        LanguageBucketNormalization::Known(bucket) => Some(bucket),
    }
}

enum LanguageBucketNormalization {
    Known(FontLanguageBucket),
    Unknown,
}

fn normalize_language_bucket(language: Option<&str>) -> LanguageBucketNormalization {
    let Some(language) = language.and_then(clean_language_tag) else {
        return LanguageBucketNormalization::Unknown;
    };
    let lower = language.to_ascii_lowercase();
    let subtags = lower
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let Some(primary) = subtags.first().copied() else {
        return LanguageBucketNormalization::Unknown;
    };

    match primary {
        "ko" | "kor" => LanguageBucketNormalization::Known(FontLanguageBucket::CjkKo),
        "ja" | "jpn" => LanguageBucketNormalization::Known(FontLanguageBucket::CjkJa),
        "zh" | "zho" | "chi" | "cmn" => {
            LanguageBucketNormalization::Known(normalize_chinese_bucket(&subtags, false))
        }
        "yue" => LanguageBucketNormalization::Known(normalize_chinese_bucket(&subtags, true)),
        "und" | "x" => LanguageBucketNormalization::Unknown,
        _ if is_well_formed_general_language_tag(&subtags) => {
            LanguageBucketNormalization::Known(FontLanguageBucket::General)
        }
        _ => LanguageBucketNormalization::Unknown,
    }
}

fn normalize_chinese_bucket(subtags: &[&str], default_traditional: bool) -> FontLanguageBucket {
    if subtags
        .iter()
        .any(|part| matches!(*part, "hant" | "tw" | "hk" | "mo" | "cht" | "traditional"))
    {
        return FontLanguageBucket::CjkZhHant;
    }
    if subtags
        .iter()
        .any(|part| matches!(*part, "hans" | "cn" | "sg" | "my" | "chs" | "simplified"))
    {
        return FontLanguageBucket::CjkZhHans;
    }
    if default_traditional {
        FontLanguageBucket::CjkZhHant
    } else {
        FontLanguageBucket::CjkZhHans
    }
}

fn normalize_explicit_locale(language: Option<&str>) -> Option<String> {
    let language = language.and_then(clean_language_tag)?;
    let lower = language.to_ascii_lowercase();
    let subtags = lower
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if !is_well_formed_general_language_tag(&subtags) {
        return None;
    }
    let mut normalized = Vec::new();
    for (index, subtag) in language
        .split('-')
        .filter(|part| !part.is_empty())
        .enumerate()
    {
        if index == 0 {
            normalized.push(subtag.to_ascii_lowercase());
        } else if subtag.len() == 4 && subtag.chars().all(|ch| ch.is_ascii_alphabetic()) {
            let mut chars = subtag.chars();
            let first = chars
                .next()
                .map(|ch| ch.to_ascii_uppercase())
                .unwrap_or_default();
            let rest = chars.as_str().to_ascii_lowercase();
            normalized.push(format!("{first}{rest}"));
        } else if (subtag.len() == 2 && subtag.chars().all(|ch| ch.is_ascii_alphabetic()))
            || (subtag.len() == 3 && subtag.chars().all(|ch| ch.is_ascii_digit()))
        {
            normalized.push(subtag.to_ascii_uppercase());
        } else {
            normalized.push(subtag.to_ascii_lowercase());
        }
    }
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.join("-"))
    }
}

fn clean_language_tag(language: &str) -> Option<String> {
    let normalized = language.trim().replace('_', "-");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn is_well_formed_general_language_tag(subtags: &[&str]) -> bool {
    let Some(primary) = subtags.first().copied() else {
        return false;
    };
    if matches!(primary, "und" | "x") {
        return false;
    }
    if !(2..=3).contains(&primary.len()) || !primary.chars().all(|ch| ch.is_ascii_alphabetic()) {
        return false;
    }
    subtags.iter().skip(1).all(|subtag| {
        (2..=8).contains(&subtag.len()) && subtag.chars().all(|ch| ch.is_ascii_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::ui_language_hint_from_windows_langid;

    #[test]
    fn windows_ui_langid_mapping_returns_only_supported_cjk_language_hints() {
        assert_eq!(
            ui_language_hint_from_windows_langid(0x0412).as_deref(),
            Some("ko-KR")
        );
        assert_eq!(
            ui_language_hint_from_windows_langid(0x0411).as_deref(),
            Some("ja-JP")
        );
        assert_eq!(
            ui_language_hint_from_windows_langid(0x0804).as_deref(),
            Some("zh-CN")
        );
        assert_eq!(
            ui_language_hint_from_windows_langid(0x0404).as_deref(),
            Some("zh-TW")
        );
        assert_eq!(
            ui_language_hint_from_windows_langid(0x0c04).as_deref(),
            Some("zh-HK")
        );
        assert_eq!(ui_language_hint_from_windows_langid(0x0409), None);
    }
}
