package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2016-02-19
 * 
 * Interface which will be implemented by both Sigmoid and Tanh neurons.
 * Allows for greater abstraction in code
 */

public interface CalcNeuron extends Neuron{
    
    /**
     * Either sigmoid or tanh
     * 
     * @return value for node
     */
    
    public double activation(double x);
    
    public double deriv(double x);
    
    public void setInputs(Neuron[] in);
    
    public void setOutputs(Neuron[] out);
    
    public void updateWeights(Propagation prop);
    
    public double getErr();
    
    public double calcErr();
    
    public void setErr(double e);
}
