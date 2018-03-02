package data;

public class NeuralNetConfiguration {
	
	private int firstHiddenNodes;
	private int secondHiddenNodes;
	private double learningRate;
	
	public NeuralNetConfiguration(int firstHiddenNodes, int secondHiddenNodes,
			double learningRate) {
		this.firstHiddenNodes = firstHiddenNodes;
		this.secondHiddenNodes = secondHiddenNodes;
		this.learningRate = learningRate;
	}

	public int getFirstHiddenNodes() {
		return firstHiddenNodes;
	}

	public int getSecondHiddenNodes() {
		return secondHiddenNodes;
	}

	public double getLearningRate() {
		return learningRate;
	}

	@Override
	public String toString() {
		StringBuffer buff = new StringBuffer();
		buff.append("first: " + firstHiddenNodes);
		buff.append(" - second: " + secondHiddenNodes);
		buff.append(" - learning rate: " + learningRate);
		return buff.toString();
	}

}
