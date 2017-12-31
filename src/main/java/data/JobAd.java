package data;

import java.util.UUID;

import de.uni_koeln.spinfo.ml_classification.data.FocusClassifyUnit;

public class JobAd extends FocusClassifyUnit{

	public JobAd(String title, String content, UUID id, String contentHTML) {
		super(content, id, contentHTML);
		super.setTitle(title);
	}

}
