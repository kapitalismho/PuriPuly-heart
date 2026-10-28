#define MyAppName "PuriPuly <3"
#define ProductionAppId "{{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"
#ifndef MyAppDirName
  #define MyAppDirName "PuriPulyHeart"
#endif
#ifndef MyAppGroupName
  #define MyAppGroupName "PuriPulyHeart"
#endif
#ifdef MyAppDataDirName
  #define MyAppDataDirNameWasExplicit
#else
  #define MyAppDataDirName "puripuly-heart"
#endif
#ifndef MyAppId
  #define MyAppId "{{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"
#endif
#if MyAppId != ProductionAppId
  #ifndef MyAppDataDirNameWasExplicit
    #error Alternate MyAppId requires an explicit isolated MyAppDataDirName.
  #endif
#endif
#if Len(Trim(MyAppDataDirName)) == 0
  #error MyAppDataDirName must be a non-empty directory name.
#endif
#if (MyAppDataDirName == ".") || (MyAppDataDirName == "..") || (Pos("\", MyAppDataDirName) > 0) || (Pos("/", MyAppDataDirName) > 0) || (Pos(":", MyAppDataDirName) > 0)
  #error MyAppDataDirName must be one safe LocalApplicationData child directory name without traversal, roots, or separators.
#endif
#define MyAppDataRoot "{localappdata}\" + MyAppDataDirName
#define MyAppVersion "2.9.1"
#define MyAppPublisher "salee"
#define MyAppURL "https://github.com/kapitalismho/PuriPuly-heart"
#ifndef MyAppExeName
  #define MyAppExeName "PuriPulyHeart.exe"
#endif
#define MyCliExeName "puripuly.exe"
#define MyOverlayExeName "PuriPulyHeartOverlay.exe"
#define MyASRWorkerExeName "PuriPulyHeartASRWorker.exe"
