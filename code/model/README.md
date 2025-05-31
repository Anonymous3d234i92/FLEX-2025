## Getting Started

1. **Download the Model**

   * Inside `code/model/`, there is a `codegen-2B` folder.
   * Download the model weights from [Salesforce/codegen-2B-multi](https://huggingface.co/Salesforce/codegen-2B-multi) and place them in the `codegen-2B` directory.

2. **Install Dependencies**

   * Make sure you have Python installed, then install the required Python packages:

   torch accelerate deepspeed peft

3. **Merge Model Files**

   * After downloading, run the following command to merge the model files:

     ```bash
     bash merge.sh
     ```
