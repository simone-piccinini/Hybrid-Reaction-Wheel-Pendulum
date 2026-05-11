class ReactionWheel:
    def __init__(self, mass: float, radius: float, friction: float):
        self.m = mass
        self.r = radius
        self.b = friction
        # Calcolo dell'inerzia (Cilindro pieno: 1/2 * m * r^2)
        self.inertia = 0.5 * self.m * (self.r**2)

    def get_torque_loss(self, velocity: float):
        """Calcola la perdita per attrito della ruota."""
        return self.b * velocity
    

class DCMotor:
    def __init__(self, K_t: float, K_e: float, R_a: float):
        self.K_t = K_t
        self.K_e = K_e
        self.R_a = R_a

    def get_torque(self, V_a: float, velocity_wheel: float) -> float:
        """
        Calcola la coppia prodotta: u = (Kt/Ra) * Va - (Kt*Ke/Ra) * omega
        """
        return (self.K_t * V_a / self.R_a) - (self.K_t * self.K_e * velocity_wheel / self.R_a)