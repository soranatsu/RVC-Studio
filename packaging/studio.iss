#ifndef StageDir
  #error StageDir is required
#endif
#ifndef ReleaseDir
  #error ReleaseDir is required
#endif
#ifndef AppVersion
  #define AppVersion "1.2.7"
#endif

[Setup]
AppId={{8E59D1AE-6A7A-45A9-A0D2-590461750F11}
AppName=RVC 声音工作台
AppVersion={#AppVersion}
AppPublisher=RVC 声音工作台
DefaultDirName={localappdata}\Programs\RVCStudio
DefaultGroupName=RVC 声音工作台
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
AppMutex=Local\RVC.MyGO.VoiceStudio
CloseApplications=no
RestartApplications=no
UninstallDisplayIcon={app}\assets\studio_icon.ico
SetupIconFile={#StageDir}\assets\studio_icon.ico
LicenseFile={#StageDir}\LICENSE
InfoBeforeFile={#StageDir}\发布说明.txt
OutputDir={#ReleaseDir}
OutputBaseFilename=RVC-Studio-{#AppVersion}-Setup
Compression=lzma2/normal
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMADictionarySize=32768
LZMANumBlockThreads=8
DiskSpanning=yes
DiskSliceSize=2000000000
SlicesPerDisk=1
WizardStyle=modern
WizardSizePercent=110
SetupLogging=yes
VersionInfoDescription=RVC 声音工作台安装程序
VersionInfoVersion={#AppVersion}.0

[Languages]
Name: "chinesesimp"; MessagesFile: "vendor\ChineseSimplified.isl"

[LangOptions]
DialogFontName=Microsoft YaHei UI
DialogFontSize=9

[Tasks]
Name: "desktopicon"; Description: "创建桌面入口"; GroupDescription: "快捷方式："
Name: "cable"; Description: "安装 VB-CABLE 通话声卡（需要管理员许可，安装后重启）"; GroupDescription: "微信等通话软件："; Check: CableMissing

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "configs\model_labels.json,*.pyc,*.pyo,\logs\*"
Source: "{#StageDir}\configs\model_labels.json"; DestDir: "{app}\configs"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{group}\RVC 声音工作台"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I ""{app}\studio_launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\studio_icon.ico"
Name: "{userdesktop}\RVC 声音工作台"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I ""{app}\studio_launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\assets\studio_icon.ico"; Tasks: desktopicon
Name: "{group}\环境检测"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I ""{app}\studio_launcher.py"" --diagnose"; WorkingDir: "{app}"; IconFilename: "{app}\assets\studio_icon.ico"
Name: "{group}\使用说明"; Filename: "{app}\使用说明.txt"
Name: "{group}\安装通话声卡"; Filename: "{app}\prerequisites\VB-CABLE\VBCABLE_Setup_x64.exe"; WorkingDir: "{app}\prerequisites\VB-CABLE"

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "-I ""{app}\studio_launcher.py"""; WorkingDir: "{app}"; Description: "打开 RVC 声音工作台"; Flags: nowait postinstall skipifsilent; Check: CanLaunch

[Code]
var
  CablePresent, RestartRequired, PrerequisiteFailed: Boolean;

function DetectCable(): Boolean;
var
  Locator, Services, Devices: Variant;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Services := Locator.ConnectServer('.', 'root\CIMV2');
    Devices := Services.ExecQuery('SELECT Name FROM Win32_SoundDevice WHERE Name = ''VB-Audio Virtual Cable'' AND Status = ''OK''');
    Result := Devices.Count > 0;
  except
    Log('Audio device detection failed; leave the driver available for manual installation.');
  end;
end;

function CableMissing(): Boolean;
begin
  Result := not CablePresent;
end;

function VCNeeded(): Boolean;
var
  Installed, Major, Minor, Build: Cardinal;
  Key: String;
begin
  Key := 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  Result := True;
  if RegQueryDWordValue(HKLM64, Key, 'Installed', Installed) and (Installed = 1) and
     RegQueryDWordValue(HKLM64, Key, 'Major', Major) and
     RegQueryDWordValue(HKLM64, Key, 'Minor', Minor) and
     RegQueryDWordValue(HKLM64, Key, 'Bld', Build) then
    Result := (Major < 14) or ((Major = 14) and ((Minor < 44) or ((Minor = 44) and (Build < 35211))));
end;

function RunPrerequisite(const FileName, Parameters: String): Boolean;
var
  Code: Integer;
begin
  Result := ShellExec('runas', FileName, Parameters, ExtractFileDir(FileName), SW_HIDE, ewWaitUntilTerminated, Code);
  if Result then begin
    Log(Format('Prerequisite %s exited with %d', [FileName, Code]));
    if (Code = 3010) or (Code = 1641) then RestartRequired := True;
    Result := (Code = 0) or (Code = 3010) or (Code = 1641);
  end else
    Log(Format('Cannot start prerequisite %s: %d', [FileName, Code]));
end;

procedure InitializeWizard();
begin
  CablePresent := DetectCable();
  Log(Format('Working VB-CABLE present: %d', [Ord(CablePresent)]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Success: Boolean;
begin
  if CurStep = ssPostInstall then begin
    if VCNeeded() then begin
      Success := RunPrerequisite(ExpandConstant('{app}\prerequisites\VC_redist.x64.exe'), '/install /quiet /norestart');
      if not Success then begin
        PrerequisiteFailed := True;
        if not WizardSilent then MsgBox('运行环境安装未完成。请重新运行安装包，并允许 Microsoft 安装程序的管理员提示。', mbError, MB_OK);
      end;
    end;
    if WizardIsTaskSelected('cable') and not DetectCable() then begin
      Success := RunPrerequisite(ExpandConstant('{app}\prerequisites\VB-CABLE\VBCABLE_Setup_x64.exe'), '-i -h');
      if Success then RestartRequired := True else begin
        if not WizardSilent then MsgBox('通话声卡安装未完成。变声器已经安装，之后可从开始菜单的“安装通话声卡”重试；右键以管理员身份运行，安装后重启。', mbInformation, MB_OK);
      end;
    end;
  end;
end;

function NeedRestart(): Boolean;
begin
  Result := RestartRequired;
end;

function CanLaunch(): Boolean;
begin
  Result := not RestartRequired and not PrerequisiteFailed;
end;
