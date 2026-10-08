; Small network bootstrap for the public 1.2.7 offline installer.
; It keeps validated volumes in a per-user cache, so a retry does not lose
; completed files. The full installer remains the authority for installation.
#define ProductVersion "1.2.7"
#define ProductName "RVC 声音工作台"
#ifndef ReleaseBase
#define ReleaseBase "https://github.com/soranatsu/RVC-Studio/releases/download/v1.2.7/"
#endif

[Setup]
AppId={{F1DAB5C3-1C8D-4C44-9C44-3E43F8C4E127}
AppName={#ProductName} 在线安装器
AppVersion={#ProductVersion}
AppPublisher={#ProductName}
AppMutex=RVCStudio.OnlineSetup.1.2.7
DefaultDirName={localappdata}\Programs\RVCStudioOnlineSetup
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
MinVersion=10.0
#ifdef OnlineSetupTest
CreateAppDir=no
DisableDirPage=yes
DisableFinishedPage=yes
OutputDir=..\..\releases\online-test
OutputBaseFilename=RVC-Studio-online-test
#else
OutputDir=..\..\releases\online-1.2.7
OutputBaseFilename=RVC-Studio-1.2.7-Online-Setup
#endif
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
SetupIconFile=..\assets\studio_icon.ico
Uninstallable=no
CreateUninstallRegKey=no
WizardStyle=modern
WizardSizePercent=110
SetupLogging=yes
VersionInfoDescription=RVC 声音工作台在线安装器
VersionInfoVersion={#ProductVersion}.0

[Languages]
Name: "chinesesimp"; MessagesFile: "online\ChineseSimplified.isl"

[Code]
{ Generated from the locally verified chunk manifest. Do not edit hashes by hand. }
#ifdef OnlineSetupTest
const
  CacheName = 'RVCStudio\\downloads\\online-test';
  AssetCount = 5;
#else
const
  CacheName = 'RVCStudio\\downloads\\1.2.7';
  AssetCount = 26;
#endif

function AssetName(Index: Integer): String;
begin
  #ifdef OnlineSetupTest
  case Index of
    0: Result := 'RVC-Studio-1.2.7-Setup.exe';
    1: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part01';
    2: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part02';
    3: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part01';
    4: Result := 'RVC-Studio-1.2.7-Setup-4.bin';
  end;
  #else
  case Index of
    0: Result := 'RVC-Studio-1.2.7-Setup.exe';
    1: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part01';
    2: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part02';
    3: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part03';
    4: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part04';
    5: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part05';
    6: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part06';
    7: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part07';
    8: Result := 'RVC-Studio-1.2.7-Setup-1.bin.part08';
    9: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part01';
    10: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part02';
    11: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part03';
    12: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part04';
    13: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part05';
    14: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part06';
    15: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part07';
    16: Result := 'RVC-Studio-1.2.7-Setup-2.bin.part08';
    17: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part01';
    18: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part02';
    19: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part03';
    20: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part04';
    21: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part05';
    22: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part06';
    23: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part07';
    24: Result := 'RVC-Studio-1.2.7-Setup-3.bin.part08';
    25: Result := 'RVC-Studio-1.2.7-Setup-4.bin';
  end;
  #endif
end;

function AssetSHA(Index: Integer): String;
begin
  #ifdef OnlineSetupTest
  case Index of
    0: Result := '6ed458523908284951545710cb5d1566037920fc692b781bd17c1ca1fb2e0f16';
    1: Result := 'dba0082ec68e72d47fa926dcc91fb9b00961ceb4342554a0c2aa567144a5a6d3';
    2: Result := 'dfd5922f5aa9441e95d21ed58bb9f2209a004912a1232e89f75f725bd37f3d4a';
    3: Result := '8c09b37123eb0d9dcec63788149e576b1614ed481ad002512d0a47a09cbf59fc';
    4: Result := 'c1db13ab836db24060490c83b83c089adb1ce9e6e1899d81e02d398cec22e6b1';
  end;
  #else
  case Index of
    0: Result := '6581e10519f96df32d9058178c58cb637fc2870f87aa9ea2271ea02f051e9287';
    1: Result := '97c1083ec11eee66c879008a7283fd38ba6aca9d67832a3689a9d1660ad75cba';
    2: Result := 'dd1a771b89fbec550bf7d8383144afcc8a3be118f0a7d3ac8a94213966e11c9c';
    3: Result := 'f83102b4af2895cf941cb911da659bdb83da5f279caeb7bee7de3d5d4b62c2a6';
    4: Result := 'c4557a60fc3ad576756b4d196f92e38b371e7e79036c49ddb679b06edaafd735';
    5: Result := '931ea6b9832067e11d6df48d693a53e9878083c46e0c6b8ddac74397c0d99322';
    6: Result := '352fcf3c0ca9fbd0cb4bd0c7706e9b3a07e2d6cd21a7289a77de90577c801125';
    7: Result := 'dee132ddbafcc007d43613d6134ec264e49850586b5cdbf031922edf75540b9c';
    8: Result := '7bed593c55c9d5055dedb3ed865ef06282256e97a351b43647e363a453235583';
    9: Result := '1aa2561d23467275c2581abd5489790d4eb66da04d7fddadd993dd17902d9e1f';
    10: Result := '52195c8a01caaa8dc28594e25e6dfa2d135c919adca7376fc0c4218f0d9afdd7';
    11: Result := '23e5b9d90a88f2e3ee83ec51df929679cfdd36a0ac8f81726dfb1d3514499ef7';
    12: Result := '61b87daee2c3fd45feae1b09c5830a05ee43dc65afee7b7f35442868c9f096aa';
    13: Result := '8e31167aebd7a6786e20ae05b65945321326c1c53e1327cf55b1c06800a86e33';
    14: Result := 'da5069b192e3fcad714d49b61a8dd76ce707d059a56f3a2def388e1ad4418f22';
    15: Result := '2ee2cfa22493a209dce8375d461b2de45e28abf217377d05d4ba49a43406d565';
    16: Result := 'fe365d82ca49914f90f5ce60b0a6bbaa25621440d71a5f6498c50335021508ce';
    17: Result := 'dafb06d9e325a882d025a9783506f135a9b16e91532732e345c4359d809dd165';
    18: Result := '48fc228fc009d1f251ee9ea3fa5996dd6f6142e3a5aa015c5518039f49621f6a';
    19: Result := '214808f48782f334265bed41ad4ff281aadc1b740376c557303522ef4d71cb3d';
    20: Result := '45879332c28c4024f2d2150375c6d60b66a51f6ec20108ee5390806c3386a54a';
    21: Result := 'd67d6367cc36073de9cec41f116b95bf6fbe24409126fe2955c816c368123e87';
    22: Result := '23a04144189ab32ebbc3cd4a0e12bcf45774c5b021c87217f1d63d8e1014edef';
    23: Result := '6af98dfd1f5eafd2ff22f9f26ebb7d56efb7ec92ddbb76adfe0108ae84489362';
    24: Result := 'e4a217578ff44ad2229b3fd5822b9907f89b60ea813a5de925cd40363a4f7749';
    25: Result := '6388be76a79be542136355f99d864d3fac7b7e8e3e2608e6e173d02e620a3ddf';
  end;
  #endif
end;

function AssetBytes(Index: Integer): Int64;
begin
  #ifdef OnlineSetupTest
  case Index of
    0: Result := 16;
    1: Result := 9;
    2: Result := 9;
    3: Result := 9;
    4: Result := 9;
  end;
  #else
  case Index of
    0: Result := 5205641;
    1: Result := 268435456;
    2: Result := 268435456;
    3: Result := 268435456;
    4: Result := 268435456;
    5: Result := 268435456;
    6: Result := 268435456;
    7: Result := 268435456;
    8: Result := 115745792;
    9: Result := 268435456;
    10: Result := 268435456;
    11: Result := 268435456;
    12: Result := 268435456;
    13: Result := 268435456;
    14: Result := 268435456;
    15: Result := 268435456;
    16: Result := 120951808;
    17: Result := 268435456;
    18: Result := 268435456;
    19: Result := 268435456;
    20: Result := 268435456;
    21: Result := 268435456;
    22: Result := 268435456;
    23: Result := 268435456;
    24: Result := 120951808;
    25: Result := 857728145;
  end;
  #endif
end;

function OriginalName(Index: Integer): String;
begin
  #ifdef OnlineSetupTest
  Result := 'RVC-Studio-1.2.7-Setup-1.bin';
  exit;
  #endif
  case Index of
    0: Result := 'RVC-Studio-1.2.7-Setup-1.bin';
    1: Result := 'RVC-Studio-1.2.7-Setup-2.bin';
    2: Result := 'RVC-Studio-1.2.7-Setup-3.bin';
  end;
end;

function OriginalSHA(Index: Integer): String;
begin
  #ifdef OnlineSetupTest
  Result := '89b009ce1f326e68e9ab2cdfde61bd6b614773b650abcf1d6c8ca6a40afcb2ba';
  exit;
  #endif
  case Index of
    0: Result := 'de7aadb3406b1d34be1dac235eec5a3a7a057e2cdf4ba4473c7367ff8ff72ed6';
    1: Result := '701fd40c89af80268405a0bd1a06a336658fb6b6def64bebba4a66c3d7c06f13';
    2: Result := 'bfd402869e37a5a6d499b50f302aaa3b50e57906ba28ad6642099486a76656bb';
  end;
end;

function OriginalBytes(Index: Integer): Int64;
begin
  #ifdef OnlineSetupTest
  Result := 18;
  exit;
  #endif
  case Index of
    0: Result := 1994793984;
    1: Result := 2000000000;
    2: Result := 2000000000;
  end;
end;

function ChunkCount(Index: Integer): Integer;
begin
  #ifdef OnlineSetupTest
  Result := 2;
  #else
  Result := 8;
  #endif
end;

function ChunkName(Index, Part: Integer): String;
begin
  #ifdef OnlineSetupTest
  if (Index = 0) and (Part = 1) then Result := 'RVC-Studio-1.2.7-Setup-1.bin.part01'
  else Result := 'RVC-Studio-1.2.7-Setup-1.bin.part02';
  #else
  Result := AssetName(1 + Index * 8 + Part - 1);
  #endif
end;

var
  StartPage: TWizardPage;
  DownloadPage: TDownloadWizardPage;
  StatusLabel, DetailLabel: TNewStaticText;
  CacheDir: String;
  Downloaded: Boolean;

function Quote(const S: String): String;
begin
  Result := '"' + S + '"';
end;

function HumanBytes(const Value: Int64): String;
begin
  if Value >= 1073741824 then
    Result := Format('%.2f GB', [Value / 1073741824])
  else if Value >= 1048576 then
    Result := Format('%.1f MB', [Value / 1048576])
  else
    Result := Format('%d B', [Value]);
end;

function VerifyFile(const Path, ExpectedSHA: String; ExpectedBytes: Int64): Boolean;
begin
  { SHA-256 is the authoritative byte and length check for these immutable assets. }
  Result := False;
  try
    Result := FileExists(Path) and
      (CompareText(GetSHA256OfFile(Path), ExpectedSHA) = 0);
  except
    Result := False;
  end;
end;

function AssembleOriginal(Index: Integer): Boolean;
var
  I, Code: Integer;
  Params, TargetPath, PartPath, TempName, TempPath: String;
begin
  TargetPath := AddBackslash(CacheDir) + OriginalName(Index);
  if VerifyFile(TargetPath, OriginalSHA(Index), OriginalBytes(Index)) then begin
    Result := True;
    exit;
  end;
  TempName := OriginalName(Index) + '.assembling';
  TempPath := AddBackslash(CacheDir) + TempName;
  DeleteFile(TempPath);
  Params := '/c copy /y /b ';
  for I := 1 to ChunkCount(Index) do begin
    PartPath := ChunkName(Index, I);
    if not VerifyFile(AddBackslash(CacheDir) + PartPath, AssetSHA(1 + Index * 8 + I - 1), AssetBytes(1 + Index * 8 + I - 1)) then begin
      Result := False;
      exit;
    end;
    if I > 1 then Params := Params + '+';
    Params := Params + Quote(PartPath);
  end;
  Params := Params + ' ' + Quote(TempName);
  if not Exec(ExpandConstant('{sys}\cmd.exe'), Params, CacheDir, SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then begin
    DeleteFile(TempPath);
    Result := False;
    exit;
  end;
  Result := VerifyFile(TempPath, OriginalSHA(Index), OriginalBytes(Index));
  if Result then begin
    DeleteFile(TargetPath);
    Result := RenameFile(TempPath, TargetPath);
  end;
  if not Result then DeleteFile(TempPath);
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  { The native download page owns the progress bar; returning False aborts safely. }
  Result := not DownloadPage.AbortedByUser;
end;

function FetchAsset(Index: Integer): Boolean;
var
  TargetPath, TempPath, URL: String;
begin
  TargetPath := AddBackslash(CacheDir) + AssetName(Index);
  URL := '{#ReleaseBase}' + AssetName(Index);
  StatusLabel.Caption := Format('正在准备第 %d/%d 个文件：%s', [Index + 1, AssetCount, AssetName(Index)]);
  if VerifyFile(TargetPath, AssetSHA(Index), AssetBytes(Index)) then begin
    DetailLabel.Caption := '已校验，跳过下载。';
    Result := True;
    exit;
  end;
  TempPath := AddBackslash(ExpandConstant('{tmp}')) + AssetName(Index);
  DeleteFile(TempPath);
  DownloadPage.Clear;
  DownloadPage.Add(URL, AssetName(Index), AssetSHA(Index));
  DownloadPage.Show;
  try
    DownloadPage.Download;
    if not VerifyFile(TempPath, AssetSHA(Index), AssetBytes(Index)) then begin
      DeleteFile(TempPath);
      Result := False;
      DetailLabel.Caption := '下载文件校验失败，临时文件已删除。请重试。';
    end else begin
      DeleteFile(TargetPath);
    Result := RenameFile(TempPath, TargetPath);
    if not Result then begin
      Result := FileCopy(TempPath, TargetPath, False);
      if Result then DeleteFile(TempPath);
    end;
      if not Result then
        DetailLabel.Caption := '无法保存校验后的文件，请检查磁盘空间或权限。';
    end;
  except
    Result := False;
    if DownloadPage.AbortedByUser then
      DetailLabel.Caption := '已取消当前下载；已完成的文件会保留，重新运行可继续。'
    else
      DetailLabel.Caption := '下载失败：' + GetExceptionMessage;
  finally
    DownloadPage.Hide;
  end;
end;

function DownloadAll(): Boolean;
var
  I, J: Integer;
begin
  ForceDirectories(CacheDir);
  #ifdef OnlineSetupTest
  for I := 0 to AssetCount - 1 do begin
    if not FetchAsset(I) then begin
      Result := False;
      exit;
    end;
  end;
  if not AssembleOriginal(0) then begin
    Result := False;
    exit;
  end;
  #else
  if not FetchAsset(0) then begin
    Result := False;
    exit;
  end;
  for I := 0 to 2 do begin
    if not VerifyFile(AddBackslash(CacheDir) + OriginalName(I), OriginalSHA(I), OriginalBytes(I)) then begin
      for J := 1 to ChunkCount(I) do begin
        if not FetchAsset(1 + I * 8 + J - 1) then begin
          Result := False;
          exit;
        end;
      end;
      StatusLabel.Caption := Format('正在重组第 %d/3 个安装分卷', [I + 1]);
      if not AssembleOriginal(I) then begin
        DetailLabel.Caption := '分卷重组或最终校验失败，未启动安装程序。';
        Result := False;
        exit;
      end;
      { Once the complete original volume is verified, release its chunks. }
      for J := 1 to ChunkCount(I) do
        DeleteFile(AddBackslash(CacheDir) + ChunkName(I, J));
    end;
  end;
  if not FetchAsset(25) then begin
    Result := False;
    exit;
  end;
  #endif
  StatusLabel.Caption := '文件全部校验完成';
  DetailLabel.Caption := '即将启动完整安装程序。缓存位置：' + CacheDir;
  Result := True;
end;

function LaunchFullSetup(): Boolean;
var
  Code: Integer;
  SetupPath: String;
begin
  #ifdef OnlineSetupTest
  RaiseException('测试安装器禁止启动完整安装程序');
  #endif
  SetupPath := AddBackslash(CacheDir) + AssetName(0);
  Result := Exec(SetupPath, '', CacheDir, SW_SHOWNORMAL, ewNoWait, Code);
  if not Result then
    MsgBox('完整安装程序启动失败，可手动运行：' + #13#10 + SetupPath, mbError, MB_OK);
end;

procedure InitializeWizard();
begin
  CacheDir := AddBackslash(ExpandConstant('{localappdata}')) + CacheName;
  StartPage := CreateCustomPage(wpWelcome, '下载完整安装包',
    '在线安装器会下载约 6.85 GB 的运行环境和模型');
  DownloadPage := CreateDownloadPage('下载完整安装包', '请等待下载完成，已完成文件会保留。', @OnDownloadProgress);
  DownloadPage.ShowBaseNameInsteadOfUrl := True;
  StatusLabel := TNewStaticText.Create(StartPage);
  StatusLabel.Parent := StartPage.Surface;
  StatusLabel.Left := ScaleX(16);
  StatusLabel.Top := ScaleY(32);
  StatusLabel.Width := StartPage.SurfaceWidth - ScaleX(32);
  StatusLabel.Caption := '点击“下一步”开始下载';
  DetailLabel := TNewStaticText.Create(StartPage);
  DetailLabel.Parent := StartPage.Surface;
  DetailLabel.Left := ScaleX(16);
  DetailLabel.Top := ScaleY(72);
  DetailLabel.Width := StartPage.SurfaceWidth - ScaleX(32);
  DetailLabel.Height := ScaleY(60);
  DetailLabel.Caption := '已下载文件可复用。缓存预留 10 GB，安装另需约 15 GB；在同一盘时建议预留 25 GB。';
  #ifdef OnlineSetupTest
  Downloaded := DownloadAll();
  #ifdef OnlineSetupExpectFailure
  if not Downloaded then
    SaveStringToFile(ExpandConstant('{localappdata}\RVCStudio\online-test-report.txt'), 'expected_failure_pass', False)
  else
    SaveStringToFile(ExpandConstant('{localappdata}\RVCStudio\online-test-report.txt'), 'expected_failure_failed', False);
  #else
  if Downloaded then
    SaveStringToFile(ExpandConstant('{localappdata}\RVCStudio\online-test-report.txt'), 'success', False)
  else
    SaveStringToFile(ExpandConstant('{localappdata}\RVCStudio\online-test-report.txt'), 'failure', False);
  #endif
  #endif
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  #ifdef OnlineSetupTest
  { The test downloads fixtures in InitializeWizard; normal silent completion
    must never interpret fixture bytes as an executable. }
  exit;
  #endif
  if CurPageID = StartPage.ID then begin
    if not Downloaded then begin
      Downloaded := DownloadAll();
      if not Downloaded then
        MsgBox('下载没有完成。重新运行此在线安装器会继续使用已有缓存。', mbError, MB_OK);
      Result := False;
      if Downloaded then begin
        if LaunchFullSetup() then WizardForm.Close;
      end;
    end else begin
      if LaunchFullSetup() then WizardForm.Close;
    end;
  end;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = StartPage.ID then begin
    WizardForm.NextButton.Caption := '开始下载';
    WizardForm.CancelButton.Caption := '取消';
  end;
end;
