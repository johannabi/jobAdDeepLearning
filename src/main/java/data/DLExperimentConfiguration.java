package data;

import java.io.File;

import de.uni_koeln.spinfo.classification.core.data.ExperimentConfiguration;
import de.uni_koeln.spinfo.classification.core.data.FeatureUnitConfiguration;
import de.uni_koeln.spinfo.classification.core.featureEngineering.featureWeighting.AbstractFeatureQuantifier;

public class DLExperimentConfiguration extends ExperimentConfiguration{
	
/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	//	private FeatureUnitConfiguration fuc;
//	private AbstractFeatureQuantifier fq;
	private NeuralNetConfiguration nnc;
	private double threshold;
//	private File dataFile;

	public DLExperimentConfiguration(FeatureUnitConfiguration fuc, AbstractFeatureQuantifier fq,
			 File dataFile) {
		super(fuc, fq, dataFile);
		this.fuc = fuc;
		this.fq = fq;
		this.dataFile = dataFile;
		this.threshold = 0.5;
		
	}
	
	public NeuralNetConfiguration getNnc() {
		return nnc;
	}

	public void setNnc(NeuralNetConfiguration nnc) {
		this.nnc = nnc;
	}
	

	
	@Override
	public String toString() {
		StringBuffer buff = new StringBuffer();
		buff.append(fuc.toString());
		if (fq != null) {
			//buff.append("_");
			buff.append(fq.getClass().getSimpleName());
		}
		
		buff.append("_thres=" + threshold);
		buff.append(nnc.toString());
		buff.append("_");
		buff.append(dataFile.getName());
		return buff.toString();
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(Double threshold) {
		this.threshold = threshold;
	}






	

}
