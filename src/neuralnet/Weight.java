package neuralnet;

/**
 * @author Alexis Varsava <av11sl@brocku.ca>
 * @version 1
 * @since 2016-03-02
 *
 * Represents the weight of a connection between two nodes in a neural network.
 */
public class Weight {

    private double value, alpha, gradient, prevGradient;

    public Weight(double v, double a) {
        value = v;
        alpha = a;
        gradient = 0.0;
        prevGradient = 0.0;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double d) {
        value = d;
    }

    public double getAlpha() {
        return alpha;
    }

    //This is only used in Delta-Bar-Delta, so the logic is unique to it
    public void setAlpha(double d) {
        if (d > 1.0) {
            alpha = 1.0;
        } else {
            alpha = d;
        }
    }

    public double getGradient() {
        return gradient;
    }

    public void setGradient(double d) {
        prevGradient = gradient;
        gradient = d;
    }

    public double getPrevGradient() {
        return prevGradient;
    }

    public String toString() {
        return Double.toString(value);
    }

}
