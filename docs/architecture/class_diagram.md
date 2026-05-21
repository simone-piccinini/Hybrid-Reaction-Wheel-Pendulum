## mermaid class diagram
classDiagram

%% ===================== PHYSICAL LAYER =====================
class DCMotor {
    +float resistance
    +float inductance
    +float torque_constant
    +float back_emf_constant
    +float rotor_inertia
    +float friction_coefficient
    +float max_voltage
    +float max_current
    +computeTorque(voltage, angular_velocity) float
    +computeCurrent(voltage, back_emf) float
}

class ReactionWheel {
    +float mass
    +float radius
    +float inertia
    +applyTorque(torque, angular_velocity) float
    +getAngularMomentum(angular_velocity) float
}

class ReactionWheelPendulum {
    +float pendulum_mass
    +float pendulum_length
    +float body_inertia
    +ReactionWheel wheel
    +DCMotor motor
    +nonlinearDynamics(state, input_torque) ndarray
    +linearize(operating_point) StateSpaceModel
}
ReactionWheelPendulum *-- ReactionWheel
ReactionWheelPendulum *-- DCMotor

%% ===================== DYNAMICS LAYER =====================
class StateSpaceModel {
    +matrix A
    +matrix B
    +matrix C
    +matrix D
    +bool is_discrete
    +float dt
    +discretize(dt) StateSpaceModel
}
ReactionWheelPendulum ..> StateSpaceModel : produces

%% ===================== CONFIGURATION =====================
class LQGConfig {
    +matrix Q_lqr
    +matrix R_lqr
    +matrix W_process
    +matrix V_measure
    +toVector() ndarray
    +fromVector(theta) LQGConfig
}

class SearchSpace {
    +int dimension
    +ndarray lower_bounds
    +ndarray upper_bounds
    +bool[] log_scale
    +sample(n) ndarray
    +clip(theta) ndarray
}

%% ===================== ESTIMATION LAYER =====================
class KalmanFilter {
    +matrix W_process
    +matrix V_measure
    +matrix P
    +vector x_hat
    +predict(u)
    +update(z)
    +getStateEstimate() ndarray
}
KalmanFilter --> StateSpaceModel

%% ===================== CONTROL LAYER =====================
class LQRController {
    +matrix Q_lqr
    +matrix R_lqr
    +matrix K
    +computeGain()
    +computeControl(x_hat) ndarray
}
LQRController --> StateSpaceModel

%% ===================== SIMULATION LAYER =====================
class SimulationEngine {
    +float dt
    +float simulation_time
    +int seed
    +run(config) SimulationResult
    +step()
    +reset(seed)
}
SimulationEngine --> ReactionWheelPendulum
SimulationEngine --> KalmanFilter
SimulationEngine --> LQRController
SimulationEngine ..> LQGConfig : consumes
SimulationEngine ..> SimulationResult : produces

class SimulationResult {
    +ndarray time
    +ndarray true_states
    +ndarray estimated_states
    +ndarray controls
    +int seed
    +bool diverged
}

%% ===================== METRICS LAYER =====================
class ObjectiveFunction {
    +float Mp_desired
    +float Ts_desired
    +float w1
    +float w2
    +computeOvershoot(result) float
    +computeSettlingTime(result) float
    +computeControlEffort(result) float
    +evaluate(result) float
}
ObjectiveFunction ..> SimulationResult : consumes

%% ===================== OPTIMIZATION LAYER =====================
class BayesianOptimizer {
    +GaussianProcess gp
    +AcquisitionFunction acquisition
    +SearchSpace space
    +Dataset data
    +int budget
    +optimize() LQGConfig
    +proposeNext() ndarray
    +evaluateCandidate(theta) float
}
BayesianOptimizer --> GaussianProcess
BayesianOptimizer --> AcquisitionFunction
BayesianOptimizer --> SearchSpace
BayesianOptimizer --> Dataset
BayesianOptimizer --> SimulationEngine
BayesianOptimizer --> ObjectiveFunction
BayesianOptimizer ..> LQGConfig : produces

class Dataset {
    +ndarray X
    +ndarray y
    +append(theta, y)
    +size() int
}

class GaussianProcess {
    +Kernel kernel
    +Dataset data
    +float noise_variance
    +fit(X, y)
    +predict(X) GPPosterior
    +logMarginalLikelihood() float
    +optimizeHyperparameters()
}
GaussianProcess --> Kernel
GaussianProcess --> Dataset
GaussianProcess ..> GPPosterior : produces

class GPPosterior {
    +ndarray mean
    +ndarray variance
}

class Kernel {
    <<interface>>
    +ndarray lengthscales
    +float signal_variance
    +covariance(x1, x2) float
    +gradient(x1, x2) ndarray
}
class SquaredExponentialARD {
    +covariance(x1, x2) float
    +gradient(x1, x2) ndarray
}
class Matern52ARD {
    +covariance(x1, x2) float
    +gradient(x1, x2) ndarray
}
Kernel <|.. SquaredExponentialARD
Kernel <|.. Matern52ARD

class AcquisitionFunction {
    <<interface>>
    +select(gp, space) ndarray
}
class EntropySearch {
    +int n_optimum_samples
    +select(gp, space) ndarray
    +optimumDistribution(gp, space) ndarray
    +expectedEntropyReduction(gp, theta) float
}
class ExpectedImprovement {
    +select(gp, space) ndarray
}
class UpperConfidenceBound {
    +float beta
    +select(gp, space) ndarray
}
AcquisitionFunction <|.. EntropySearch
AcquisitionFunction <|.. ExpectedImprovement
AcquisitionFunction <|.. UpperConfidenceBound

%% ===================== EXPERIMENT LAYER =====================
class ExperimentManager {
    +string git_hash
    +int seed
    +loadConfig(path) LQGConfig
    +saveResults(result)
    +logMetrics(metrics)
}
ExperimentManager --> BayesianOptimizer