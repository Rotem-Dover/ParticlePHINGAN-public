//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
/// \file TimedProcess.hh
/// \brief G4WrapperProcess that times each GPIL/DoIt call of the wrapped
///        process into ProcessTimer. Pure forwarding otherwise: the wrapped
///        process sees exactly the calls it would have seen unwrapped.
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef TimedProcess_h
#define TimedProcess_h 1

#include "G4WrapperProcess.hh"

class ProcessTimer;

class TimedProcess : public G4WrapperProcess
{
public:
  /// Takes ownership semantics of G4WrapperProcess::RegisterProcess; the
  /// wrapper adopts the wrapped process's name, type and sub-type so
  /// G4PhysicsListHelper ordering and every GetProcessName() consumer see
  /// the original identity ("hIoni").
  explicit TimedProcess(G4VProcess* wrapped);

  G4double AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                 G4double previousStepSize,
                                                 G4double currentMinimumStep,
                                                 G4double& proposedSafety,
                                                 G4GPILSelection* selection) override;

  G4double PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                G4double previousStepSize,
                                                G4ForceCondition* condition) override;

  G4VParticleChange* AlongStepDoIt(const G4Track& track, const G4Step& step) override;
  G4VParticleChange* PostStepDoIt (const G4Track& track, const G4Step& step) override;

private:
  ProcessTimer& fTimer;   // cached: no Instance() guard check on the hot path
};

#endif
