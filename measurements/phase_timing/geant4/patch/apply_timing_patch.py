#!/usr/bin/env python3
"""Install the hIoni per-phase timing shim into a StepsGenerator (TestEm5 fork)
checkout. Idempotent; asserts on every anchor so a drifted copy fails loud.

  python3 apply_timing_patch.py /path/to/StepsGenerator

Copies ProcessTimer.{hh,cc} + TimedProcess.{hh,cc} beside this script into
include/ and src/, then wires:
  * PhysListEmStandard.cc  -- wrap proton/pi hIoni in TimedProcess when enabled
  * PhysicsListMessenger   -- /testem/phys/timeProcesses <bool> (PreInit)
  * RunAction.cc           -- BeginRun/EndRun/Report around the event loop
CMake GLOBs src/*.cc, so re-run `cmake .` in the build dir before `make`.
"""
import pathlib, shutil, sys

root = pathlib.Path(sys.argv[1]).resolve()
here = pathlib.Path(__file__).resolve().parent
inc, src = root / "include", root / "src"
assert (src / "PhysListEmStandard.cc").exists(), f"not a StepsGenerator tree: {root}"

for f in ("ProcessTimer.hh", "TimedProcess.hh"):
    shutil.copy(here / f, inc / f)
for f in ("ProcessTimer.cc", "TimedProcess.cc"):
    shutil.copy(here / f, src / f)

def edit(path, pairs, marker):
    s = path.read_text()
    if marker in s:
        print(f"  {path.name}: already patched")
        return
    for old, new in pairs:
        assert old in s, f"{path.name}: anchor not found:\n{old}"
        s = s.replace(old, new, 1)
    path.write_text(s)
    print(f"  {path.name}: patched")

# --- physics list ---------------------------------------------------------
edit(src / "PhysListEmStandard.cc", [
    ('#include "G4hIonisation.hh"\n',
     '#include "G4hIonisation.hh"\n#include "TimedProcess.hh"\n#include "ProcessTimer.hh"\n'),
    ('      ph->RegisterProcess(new G4hIonisation(), particle);',
     '      // /testem/phys/timeProcesses true wraps hIoni in a TimedProcess\n'
     '      // (per-phase wall-time buckets, see ProcessTimer); pure forwarding,\n'
     '      // same name/type/subtype, so ordering and step labels are unchanged.\n'
     '      G4VProcess* hIoni = new G4hIonisation();\n'
     '      if (ProcessTimer::Instance().Enabled()) hIoni = new TimedProcess(hIoni);\n'
     '      ph->RegisterProcess(hIoni, particle);'),
], marker="TimedProcess(hIoni)")

# --- messenger --------------------------------------------------------------
edit(inc / "PhysicsListMessenger.hh", [
    ("class G4UIcmdWithAString;", "class G4UIcmdWithAString;\nclass G4UIcmdWithABool;"),
    ("  G4UIcmdWithAString*        fListCmd;",
     "  G4UIcmdWithAString*        fListCmd;\n  G4UIcmdWithABool*          fTimeCmd;"),
], marker="fTimeCmd")
edit(src / "PhysicsListMessenger.cc", [
    ('#include "G4UIcmdWithAString.hh"',
     '#include "G4UIcmdWithAString.hh"\n#include "G4UIcmdWithABool.hh"\n#include "ProcessTimer.hh"'),
    ("  fListCmd->SetToBeBroadcasted(false);",
     "  fListCmd->SetToBeBroadcasted(false);\n\n"
     '  fTimeCmd = new G4UIcmdWithABool("/testem/phys/timeProcesses",this);\n'
     '  fTimeCmd->SetGuidance("Wrap hIoni in a timing shim (per-phase wall time,");\n'
     '  fTimeCmd->SetGuidance("printed at end of run). Serial run manager only.");\n'
     '  fTimeCmd->SetParameterName("on",true);\n'
     "  fTimeCmd->SetDefaultValue(true);\n"
     "  fTimeCmd->AvailableForStates(G4State_PreInit);\n"
     "  fTimeCmd->SetToBeBroadcasted(false);"),
    ("  delete fListCmd;", "  delete fTimeCmd;\n  delete fListCmd;"),
    ("    { fPhysicsList->AddPhysicsList(newValue); }",
     "    { fPhysicsList->AddPhysicsList(newValue); }\n"
     "  else if( command == fTimeCmd )\n"
     "    { ProcessTimer::Instance().SetEnabled(fTimeCmd->GetNewBoolValue(newValue)); }"),
], marker="fTimeCmd")

# --- run action -------------------------------------------------------------
edit(src / "RunAction.cc", [
    ('#include "RunAction.hh"',
     '#include "RunAction.hh"\n#include "ProcessTimer.hh"\n#include "G4RunManager.hh"'),
    ("void RunAction::BeginOfRunAction(const G4Run*)\n{",
     "void RunAction::BeginOfRunAction(const G4Run*)\n{\n"
     "  if (isMaster) ProcessTimer::Instance().BeginRun();"),
    ("void RunAction::EndOfRunAction(const G4Run*)\n{",
     "void RunAction::EndOfRunAction(const G4Run*)\n{\n"
     "  if (isMaster) {\n"
     "    ProcessTimer::Instance().EndRun();\n"
     "    ProcessTimer::Instance().Report(\n"
     "        G4cout, G4RunManager::GetRunManager()->GetCurrentRun()->GetNumberOfEvent());\n"
     "  }"),
], marker="ProcessTimer::Instance().BeginRun()")
print("done")
