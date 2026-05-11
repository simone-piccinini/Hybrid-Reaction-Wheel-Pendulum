from dataclasses import dataclass
import yaml
from dataclasses import dataclass
import yaml

@dataclass(frozen=True)
class PendulumParams:
    m: float
    l: float
    b: float

@dataclass(frozen=True)
class WheelParams:
    m: float
    r: float
    b: float

@dataclass(frozen=True)
class MotorParams: 
    K_t: float
    K_e: float
    R_a: float

@dataclass(frozen=True)
class PhysicalConfig:
    g: float
    pendulum: PendulumParams
    wheel: WheelParams
    motor: MotorParams

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)['physics']
        
        # Mappatura manuale delle sezioni
        p = PendulumParams(**data['pendulum'])
        w = WheelParams(**data['wheel'])
        m = MotorParams(**data['motor'])
        
        return cls(g=data['g'], pendulum=p, wheel=w, motor=m)