import tensorflow as tf
from tensorflow.keras import models, layers, metrics
  

class pod_deepnet(models.Model):
    def __init__(self, branch_1, *args, **kwargs):
        super(pod_deepnet, self).__init__(*args, **kwargs)
        self.branch_1 = branch_1
        self.loss_tracker_1 = metrics.Mean(name="loss (u)")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error (u)")

    @tf.function
    def call(self, inputs):
        b_u = self.branch_1(inputs)
        return b_u
    
    
    def compile(self, optimizer_1, loss_1):
        super(pod_deepnet, self).compile()
        self.optimizer_1 = optimizer_1  
        self.loss_1 = loss_1            
   
    @tf.function
    def train_step(self, data):

        [inputs], [b_u_outputs] = data

        with tf.GradientTape(persistent=True) as tape:
            b_u = self(inputs, training=True)
            loss_u = self.loss_1(b_u_outputs, b_u)


        grads_u = tape.gradient(loss_u, self.branch_1.trainable_variables)
        self.optimizer_1.apply_gradients(zip(grads_u, self.branch_1.trainable_variables))
        error_u = tf.norm(b_u_outputs - tf.cast(b_u,tf.float64), axis=-1) / tf.norm(b_u_outputs, axis=-1) * 100

        self.loss_tracker_1.update_state(loss_u)
        self.loss_tracker_2.update_state(error_u)

        return {"loss (u)": self.loss_tracker_1.result(), "l2-norm error (u)": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):

        [inputs], [b_u_outputs] = data

        b_u = self(inputs, training=False)
        loss_u = self.loss_1(b_u_outputs, b_u)
        error_u = tf.norm(b_u_outputs - tf.cast(b_u,tf.float64), axis=-1) / tf.norm(b_u_outputs, axis=-1) * 100

        self.loss_tracker_1.update_state(loss_u)
        self.loss_tracker_2.update_state(error_u)

        return {"loss (u)": self.loss_tracker_1.result(), "l2-norm error (u)": self.loss_tracker_2.result()}
    
