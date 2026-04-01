import  numpy  as np  
import  matplotlib.pyplot as plt 
import  tensorflow as tf 
from  tensorflow import  keras 
from PIL import Image

@tf.keras.utils.register_keras_serializable(package="Custom")
class patchEmbedd(tf.keras.layers.Layer) :
    def __init__(self,  patch_size,**kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def get_config(self):
        config =super().get_config()
        config.update({
            "patch_size": self.patch_size
        })
        return config



    def call(self,  images) : 
        batch_size = tf.shape(images)[0]
        img_h = images.shape[1]
        img_w = images.shape[2]
        n_h = img_h // self.patch_size
        n_w = img_w // self.patch_size
        patches= tf.image.extract_patches(
            images =images,  
            sizes = [1,self.patch_size,self.patch_size,1],  
            padding="VALID", 
            rates=[1,1,1,1],
            strides=[1,self.patch_size,self.patch_size,1],
        
        )
        patches = tf.reshape(patches,  [batch_size,-1,patches.shape[-1]])
     
        return patches,n_h,n_w
    

   
def test_embd(size_patch :int=32, im_path :  str = r"archive\Images\3759230208_1c2a492b12.jpg") :
       encoder = patchEmbedd(size_patch)
       im_test_path = im_path
       image_test =  Image.open(im_test_path)
       image_test_tensor = tf.convert_to_tensor(image_test)
   

       n_h = image_test_tensor.shape[0]//size_patch
       n_w =  image_test_tensor.shape[1]//size_patch
       image_test_tensor = tf.expand_dims(image_test_tensor, axis=0) # comme on doit passer un batch on  expand la dim
       patches = encoder(image_test_tensor)

       patches_ = []

       for i, patch in  enumerate(patches[0]) :
         assert(i<n_h*n_w)
         patch_im = tf.reshape(patch, [size_patch,size_patch,3])
         patches_.append(patch_im)
  
       return patches_, n_h, n_w
def plot_patch(patches :  tf.Tensor , n_h :int, n_w :int) :
    plt.figure(figsize=(10,8))
    for i, patch in enumerate(patches) : 
        assert(i<n_h*n_w)
        ax= plt.subplot(n_h,n_w,i+1)
        ax.imshow(patch.numpy().astype("uint8"))
        ax.axis("off")
    plt.tight_layout()
    plt.show()

 
       
if __name__ == '__main__' :
    #patches, n_h, n_w= test_embd()
    #plot_patch(patches, n_h, n_w)
    print("lol")
   