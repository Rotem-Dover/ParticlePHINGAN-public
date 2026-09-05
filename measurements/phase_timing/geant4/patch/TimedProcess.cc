//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
/// \file TimedProcess.cc
/// \brief Implementation of TimedProcess (see header).
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "TimedProcess.hh"
#include "ProcessTimer.hh"

namespace {
inline ProcessTimer::ticks Since(ProcessTimer::ticks t0)
{
  return ProcessTimer::Now() - t0;
}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

TimedProcess::TimedProcess(G4VProcess* wrapped)
  : G4WrapperProcess("", wrapped->GetProcessType()), fTimer(ProcessTimer::Instance())
{
  // G4WrapperProcess::RegisterProcess APPENDS the wrapped name to ours and
  // copies the type, so start from "" to end up as exactly "hIoni"; the
  // sub-type is not copied by the base, so do it here (ordering lookup).
  RegisterProcess(wrapped);
  SetProcessSubType(wrapped->GetProcessSubType());
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double TimedProcess::AlongStepGetPhysicalInteractionLength(
    const G4Track& track, G4double previousStepSize, G4double currentMinimumStep,
    G4double& proposedSafety, G4GPILSelection* selection)
{
  const ProcessTimer::ticks t0 = ProcessTimer::Now();
  const G4double r = pRegProcess->AlongStepGetPhysicalInteractionLength(
      track, previousStepSize, currentMinimumStep, proposedSafety, selection);
  fTimer.AddAlongGPIL(Since(t0));
  return r;
}

G4double TimedProcess::PostStepGetPhysicalInteractionLength(
    const G4Track& track, G4double previousStepSize, G4ForceCondition* condition)
{
  const ProcessTimer::ticks t0 = ProcessTimer::Now();
  const G4double r = pRegProcess->PostStepGetPhysicalInteractionLength(
      track, previousStepSize, condition);
  fTimer.AddPostGPIL(Since(t0));
  return r;
}

G4VParticleChange* TimedProcess::AlongStepDoIt(const G4Track& track, const G4Step& step)
{
  const ProcessTimer::ticks t0 = ProcessTimer::Now();
  G4VParticleChange* r = pRegProcess->AlongStepDoIt(track, step);
  fTimer.AddAlongDoIt(Since(t0));
  return r;
}

G4VParticleChange* TimedProcess::PostStepDoIt(const G4Track& track, const G4Step& step)
{
  const ProcessTimer::ticks t0 = ProcessTimer::Now();
  G4VParticleChange* r = pRegProcess->PostStepDoIt(track, step);
  fTimer.AddPostDoIt(Since(t0));
  return r;
}
