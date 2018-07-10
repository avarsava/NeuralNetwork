package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2016-02-19
 * 
 * Either a hidden or an output node, uses a tanh function to return its
 * value.
 */
public class TanhNeuron implements CalcNeuron {

    private Neuron[] inputs;
    private Neuron[] outputs;
    private double err;
    private double lastSum;
    private int id;
    private Weight[][] inputWAM;
    private Weight[][] outputWAM;

    /**
     * Constructor. Used to initialize an output neuron.
     * 
     * @param n ID to assign to neuron, used for matrix lookup
     * @param newWAM matrix which describes weights between this neuron and its
     * inputs.
     */
    public TanhNeuron(int n, Weight[][] newWAM) {
        this.outputs = null;
        id = n;
        inputWAM = newWAM;
    }
    
    /**
     * Constructor. Used to initialize hidden neurons.
     * 
     * @param n ID to assign to neuron, used for matrix lookup
     * @param newWAM Matrix which describes weights between this neuron and its
     * inputs
     * @param newWAM2 Matrix which describes weights between this neuron 
     * and its outputs.
     */
    public TanhNeuron(int n, Weight[][] newWAM, Weight[][] newWAM2){
        this.outputs = null;
        id = n;
        inputWAM = newWAM;
        outputWAM = newWAM2;
    }

    /**
     * Setter for inputs
     * 
     * @param in list of neurons which to set inputs
     */
    public void setInputs(Neuron[] in) {
        inputs = in;
    }

    public Weight[][] getInputWAM(){
        return inputWAM;
    }

    public Weight[][] getOutputWAM(){
        return outputWAM;
    }
    
    /**
     * Setter for outputs
     * 
     * @param out list of neurons which to set outputs
     */
    public void setOutputs(Neuron[] out) {
        outputs = out;
    }

    /**
     * Calculates new value. Value is equal to the summation of the inputs
     * run through the sigmoid function.
     * 
     * @return new value
     */
    @Override
    public double getValue() {
        return activation(summation());
    }

    /**
     * To avoid recalculating at inopportune times, one has the option of 
     * calculating the sigmoid of a previously calculated summation.
     * 
     * @return sigmoid of previously calculated summation
     */
    public double getSavedValue() {
        return activation(lastSum);
    }

    private double summation() {
        double sum = 0;
        //for all Valuables in inputs
        for (Neuron v : inputs) {
            //add to sum: weight of path (from WAM) * output
            sum += inputWAM[v.getID()][id].getValue() * v.getValue();
        }
        lastSum = sum;
        return lastSum;
    }

    @Override
    public double activation(double x) {
        return Math.tanh(x);
    }

    /**
     * Derivative of the sigmoid function, courtesy of Wolfram Alpha.
     * 
     * @param x value with which to calculate function
     * @return final value
     */
    @Override
    public double deriv(double x) {
        return Math.pow((1/Math.cosh(x)), 2.0);
    }

    /**
     * Setter for error value
     * 
     * @param e value which to set err
     */
    public void setErr(double e) {
        err = e;
    }

    /**
     * Getter for error value
     * 
     * @return err
     */
    @Override
    public double getErr() {
        return err;
    }

    /**
     * Error is calculated as the sum of all weights following neuron times
     * all error values of following neurons.
     * 
     * @return calculated error for this node.
     */
    public double calcErr() {
        double ret = 0;
        if (outputs != null) {
            for (Neuron o : outputs) {
                ret += outputWAM[id][o.getID()].getValue() * o.getErr();
            }
        }
        return ret;
    }

    /**
     * Updates the weights between the inputs of this neuron, and itself.
     * 
     * New weight is calculated as the current weight between the neurons,
     * plus the learning rate times the error at this neuron, times the 
     * derivative of the sigmoid of the summation at this neuron, times the
     * value of the preceding neuron.
     */
    public void updateWeights(Propagation prop) {
        TanhNeuron n2;
        for (Neuron n : inputs) {
            inputWAM[n.getID()][id].setValue(inputWAM[n.getID()][id].getValue()
                    + (inputWAM[n.getID()][id].getAlpha()
                            * getErr() * deriv(lastSum) * n.getSavedValue()));
            //update gradients if we're using Delta-Bar-Delta
            if(prop instanceof DeltaBarDelta && n instanceof SigmoidNeuron){
                n2 = (TanhNeuron)n;
                for(Weight[] w : n2.getInputWAM()){
                    w[n2.getID()].setGradient(deriv(lastSum));
                }
            }
        }
    }

    /**
     * Getter for id
     * 
     * @return id
     */
    @Override
    public int getID() {
        return id;
    }

    /**
     * Initializes output neurons and joins them with their weighted adjacency
     * matrix.
     * 
     * @param neurons list of output neurons
     * @param weights weighted adjacency matrix describing connections between
     * hidden neurons and output neurons
     */
    public static void initNeurons(Neuron[] neurons, Weight[][] weights) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new TanhNeuron(i, weights);
        }
    }
    
    /**
     * Initializes hidden neurons and joins them with the weighted adjacency
     * matrices describing the connections on both sides of them.
     * 
     * @param neurons list of hidden neurons
     * @param weights weighted adjacency matrix describing the connections to 
     * the left of the neurons
     * @param weights2 weights adjacency matrix describing the connections to 
     * the right of the neurons
     */
    public static void initNeurons(Neuron[] neurons, Weight[][] weights, 
            Weight[][] weights2) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new TanhNeuron(i, weights, weights2);
        }
    }
}
