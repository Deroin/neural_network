package de.dero.neural_network;

import java.io.Serializable;

public class Weight implements Serializable {
    private static final long serialVersionUID = 3;
    double value;
    double hessian  = 0;
}
