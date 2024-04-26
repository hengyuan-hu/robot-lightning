import numpy as np
import zerorpc
import pyrallis

from .polymetis_controller import PolyMetisController, PolyMetisControllerConfig


def main():
    ctrl_cfg = pyrallis.parse(config_class=PolyMetisControllerConfig)
    controller = PolyMetisController(ctrl_cfg)
    controller.hello()
    s = zerorpc.Server(controller)
    s.bind("tcp://0.0.0.0:4242")
    s.run()


if __name__ == "__main__":
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    main()
