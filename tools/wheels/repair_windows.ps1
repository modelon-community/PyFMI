# delvewheel needs --add-path because FMIL and the UCRT64 runtime are not on
# the system DLL search path inside cibuildwheel's worker process.
param([string]$Wheel, [string]$DestDir)
delvewheel repair `
    --add-path "C:\msys64\ucrt64\bin;C:\deps\bin;C:\deps\lib" `
    -w $DestDir `
    $Wheel
