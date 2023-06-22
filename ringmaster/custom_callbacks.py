import tempfile
from PIL import Image
from openbabel import pybel
from pytorch_lightning.callbacks import Callback


class GenerateOnValidationCallback(Callback):
   def on_validation_end(self, trainer, pl_module):       
        wandb_logger = trainer.logger
        smiles_list = []
        for _ in range(9):
            smiles_list.append(pl_module.denoise_sample(1, 100, skip_special_tokens=True)[0])
        wimg_list = []
        for _, smiles in enumerate(smiles_list):
            try:
                mol = pybel.readstring('smi', smiles)
                with tempfile.NamedTemporaryFile(suffix='png', delete=True) as tmp_file:
                    mol.draw(show=False, filename=tmp_file.name)
                    pil_img = Image.open(tmp_file.name)
                    wimg_list.append(pil_img)
            except:
                continue
        if wimg_list:
            wandb_logger.log_image(key='diffused_smiles',images=wimg_list)

