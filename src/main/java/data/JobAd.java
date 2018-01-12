package data;

import java.util.UUID;

import org.nd4j.linalg.api.ndarray.INDArray;

import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;

public class JobAd extends FocusClassifyUnit{
	
	private INDArray array;

	public JobAd(String title, String content, UUID id, String contentHTML) {
		super(content, id, contentHTML);
		super.setTitle(title);
	}

	public INDArray getArray() {
		return array;
	}

	public void setArray(INDArray array) {
		this.array = array;
	}
	
	
	
	

}
