package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2016-02-19
 * 
 * Allows for switching out the propagation algorithm used.
 */
public interface Propagation {
    
    public void doTraining(NeuralNet nn);
}
