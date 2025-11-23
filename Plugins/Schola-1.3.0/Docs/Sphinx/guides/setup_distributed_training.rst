Distributed Training with RayCluster
====================================

In this guide, we will walk through the process of setting up a Ray cluster on MicroK8s with multiple nodes and launching a Schola training script on it. We will cover the necessary installations and configurations for MicroK8s, Docker, and Ray. It is important to note that this is not the only way to set up a Ray cluster or launch training scripts, and the configurations can be customized based on your specific requirements. However, this guide provides a starting point for distributed training using Ray on a local Kubernetes cluster.

Install Prerequisites
---------------------

Before we begin, ensure that you have the following prerequisites installed on your system:

.. tabs::

   .. group-tab:: Linux

      - |ubuntu_version|_ |ubuntu_version_exact|
      - `Docker <https://docs.docker.com/engine/install/ubuntu/>`_ (Ensure Docker is installed and running)
      - `MicroK8s <https://microk8s.io/docs/>`_ (A lightweight Kubernetes distribution)
      - `Ray <https://docs.ray.io/en/latest/installation.html>`_ (A framework for building and running distributed applications)

      .. note::
         Ensure that you have ``sudo`` privileges to install and configure the above tools.

Setting Up Docker  
-----------------  
  
#. **Uninstall Conflicting Docker Packages if Necessary**:  
  
   .. code-block:: bash  
  
      sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras  
      sudo rm -rf /var/lib/docker  
      sudo rm -rf /var/lib/containerd  
      for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done  
  
#. **Install Docker**:  
  
   .. code-block:: bash  
  
      sudo apt-get update  
      sudo apt-get install ca-certificates curl  
      sudo install -m 0755 -d /etc/apt/keyrings  
      sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc  
      sudo chmod a+r /etc/apt/keyrings/docker.asc  
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null  
      sudo apt-get update  
      sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin  
  
#. **Post Docker Installation**:  
  
   .. code-block:: bash  
  
      sudo groupadd docker  
      sudo usermod -aG docker $USER  
      newgrp docker  
      docker run hello-world  
      sudo systemctl enable docker.service  
      sudo systemctl enable containerd.service  
  
#. **Configure Docker Registry**:  
  
   .. code-block:: bash  
  
      docker run -d -p <your_registry_ip>:32000:5000 --name registry registry:2  
      cat <<EOF | sudo tee /etc/docker/daemon.json  
      {  
        "insecure-registries": ["<your_registry_ip>:32000"]  
      }  
      EOF  
      sudo systemctl restart docker  
      docker start registry  

Setting Up MicroK8s  
-------------------  
  
#. **Uninstall Existing MicroK8s if Necessary**:  
  
   .. code-block:: bash  
  
      if command -v microk8s &> /dev/null; then  
        sudo microk8s reset  
        sudo snap remove microk8s  
      fi  
      sudo rm -rf /var/snap/microk8s/current  
  
#. **Install MicroK8s**:  
  
   .. code-block:: bash  
  
      sudo snap install microk8s --classic  
      microk8s status --wait-ready  

#. **Add Your User to the MicroK8s Group**:  
  
   .. code-block:: bash  
  
      sudo usermod -a -G microk8s $USER  
      sudo chown -f -R $USER ~/.kube  
  
   .. note::  
      Log out and back in to apply the group changes.  
  
#. **Enable Necessary MicroK8s Services**:  
  
   .. code-block:: bash  
  
      microk8s enable dns storage registry  
  
5. **Configure MicroK8s to Use Local Docker Registry**:  
  
   .. code-block:: bash  
  
      sudo mkdir -p /var/snap/microk8s/current/args/certs.d/<your_registry_ip>:32000  
      cat <<EOF | sudo tee /var/snap/microk8s/current/args/certs.d/<your_registry_ip>:32000/hosts.toml  
      server = "http://<your_registry_ip>:32000"  
      [host."http://<your_registry_ip>:32000"]  
      capabilities = ["pull", "resolve"]  
      EOF  
      sudo systemctl restart docker  
      sudo snap stop microk8s  
      sudo snap start microk8s  
      microk8s status --wait-ready  
  
6. **Test MicroK8s Setup**:  
  
   .. code-block:: bash  
  
      docker start registry  
      docker pull hello-world  
      docker tag hello-world <your_registry_ip>:32000/hello-world  
      docker push <your_registry_ip>:32000/hello-world  
      microk8s kubectl create deployment hello-world --image=<your_registry_ip>:32000/hello-world  
      sleep 2  
      microk8s kubectl get deployments  
  
7. **Add Nodes to MicroK8s Cluster**:  
  
   To add a new node, first install MicroK8s on the new machine:  
  
   .. code-block:: bash  
  
      sudo snap install microk8s  
  
   Then, on the main node, generate the join command:  
  
   .. code-block:: bash  
  
      join_command=$(microk8s add-node | grep 'microk8s join' | grep 'worker')  
  
   Run the join command on the new node:  
  
   .. code-block:: bash  
  
      microk8s join <main_node_ip>:25000/<token>  
  
   Update configuration files on the new node to pull images from the local registry (Follow steps 5 in this section and 4 from Docker setup):  
  
   - Update ``/var/snap/microk8s/current/args/certs.d/<your_registry_ip>:32000/hosts.toml``  
   - Update ``/etc/docker/daemon.json``  
   - Restart the container runtime:  
  
     .. code-block:: bash  
  
        sudo systemctl restart docker  
        sudo snap stop microk8s  
        sudo snap start microk8s  

Building and Deploying the Docker Image  
---------------------------------------  
  
#. **Create a Dockerfile**:  
  
   Use the following Dockerfile as a reference to build your Docker image:  
  
   .. code-block:: docker  
  
      FROM rayproject/ray:latest-py39  
      COPY . ./python  
      RUN sudo apt-get update && cd python && python -m pip install --upgrade pip && \  
          pip install .[all] && pip install --upgrade numpy==1.26 && \  
          pip install --upgrade ray==2.36 && pip install tensorboard  
      WORKDIR ./python  
  
#. **Build the Docker Image**:  
  
   Navigate to the directory containing your Dockerfile and run:  
  
   .. code-block:: bash  
  
      docker build --no-cache -t <image_name>:<tag> .  
  
#. **Push the Docker Image to the MicroK8s Registry**:  
  
   .. code-block:: bash  
  
      docker tag <image_name>:<tag> localhost:32000/<image_name>:<tag>  
      docker push localhost:32000/<image_name>:<tag>  
   

Configuring the Ray Cluster  
---------------------------  
  
#. **Install KubeRay Operator**:  
  
   .. code-block:: bash  
  
      microk8s helm repo add kuberay https://ray-project.github.io/kuberay-helm/  
      microk8s helm repo update  
      microk8s helm install kuberay-operator kuberay/kuberay-operator --version 1.1.1  
      sleep 2  
      microk8s kubectl get pods -o wide  
  
#. **Create a RayCluster Config File**:  
  
   You may use the following YAML configuration as reference when defining your Ray cluster:  
  
   .. raw:: html

      <details>
      <summary><a><code>raycluster.yaml</code></a></summary>

   .. code-block:: yaml  
  
      apiVersion: ray.io/v1  
      kind: RayCluster  
      metadata:  
        annotations:  
          meta.helm.sh/release-name: raycluster  
          meta.helm.sh/release-namespace: default  
        labels:  
          app.kubernetes.io/instance: raycluster  
          app.kubernetes.io/managed-by: Helm  
          helm.sh/chart: ray-cluster-1.1.1  
        name: raycluster-kuberay  
        namespace: default  
      spec:  
        headGroupSpec:  
          rayStartParams:  
            dashboard-host: 0.0.0.0  
          serviceType: ClusterIP  
          template:  
            spec:  
              containers:  
              - image: <cluster IP>:32000/ScholaExamples:registry  
                imagePullPolicy: Always  
                name: ray-head  
                resources:  
                  limits:  
                    cpu: "8"  
                    memory: 48Gi  
                  requests:  
                    cpu: "2"  
                    memory: 16Gi  
              volumes:  
              - emptyDir: {}  
                name: log-volume  
        workerGroupSpecs:  
        - groupName: workergroup  
          maxReplicas: 5  
          minReplicas: 3  
          replicas: 3  
          template:  
            metadata:  
              labels:  
                app: worker-pod  
            spec:  
              containers:  
              - image: <cluster IP>:32000/ScholaExamples:registry  
                imagePullPolicy: Always  
                name: ray-worker  
                resources:  
                  limits:  
                    cpu: "8"  
                    memory: 32Gi  
                  requests:  
                    cpu: "3"  
                    memory: 6Gi  
              volumes:  
              - emptyDir: {}  
                name: log-volume  
            imagePullSecrets: []
            nodeSelector: {}
            tolerations: []
            volumes:
            - emptyDir: {}
               name: log-volume
      workerGroupSpecs:
      - groupName: workergroup
         maxReplicas: <max number of worker pods>
         minReplicas: <min number of worker pods>
         numOfHosts: 1
         rayStartParams: {}
         replicas: 3
         template:
            metadata:  
            labels:  
               app: worker-pod  
            spec:
            affinity: 
               podAntiAffinity:
                  preferredDuringSchedulingIgnoredDuringExecution: 
                  - weight: 100
                  podAffinityTerm:
                     labelSelector: 
                        matchExpressions:  
                        - key: app  
                           operator: In  
                           values:  
                              - worker-pod  
                     topologyKey: "kubernetes.io/hostname"
            containers:
            - image: <cluster IP>:32000/ScholaExamples:registry
               imagePullPolicy: Always
               name: ray-worker
               resources:
                  limits:
                     cpu: "<num cores>"
                     memory: <memory to use>Gi
                  requests:
                     cpu: "<num cores per worker>"
                     memory: <memory per worker>Gi
               securityContext: {}
            imagePullSecrets: []
            nodeSelector: {}
            tolerations: []
            volumes:
            - emptyDir: {}
               name: log-volume

   .. raw:: html

      </details>

#. **Deploy RayCluster**:  
  
   .. code-block:: bash  
  
      microk8s helm install raycluster kuberay/ray-cluster --version 1.1.1  
      sleep 2  
      microk8s kubectl get rayclusters  
      sleep 2  
      microk8s kubectl get pods --selector=ray.io/cluster=raycluster-kuberay  
      sleep 2  
      echo "Ray Cluster Pods:"  
      microk8s kubectl get pods -o wide  
  
#. **Verify Ray Cluster Setup**:  
  
   .. code-block:: bash  
  
      export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)  
      echo $HEAD_POD  
  
      get_head_pod_status() {  
          HEAD_POD=$(microk8s kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)  
          microk8s kubectl get pods | grep $HEAD_POD | awk '{print $3}'  
      }  
  
      head_pod_status=$(get_head_pod_status)  
      while [ "$head_pod_status" != "Running" ]; do  
          echo "Current head pod ($HEAD_POD) status: $head_pod_status. Waiting for 'Running'..."  
          sleep 2  
          head_pod_status=$(get_head_pod_status)  
      done  
  
      kubectl exec -it $HEAD_POD -- python -c "import ray; ray.init(); print(ray.cluster_resources())"  

Deploying the Ray cluster
-------------------------

Apply the configuration to your MicroK8s cluster:  
  
.. code-block:: bash  

   microk8s kubectl apply -f raycluster.yaml


Launching the Training Script  
-----------------------------  
  
#. **Execute the Training Script on the Ray Cluster**:  
  
   The following command is used to launch the training script on the Ray cluster:  
  
   .. code-block:: bash  
  
      microk8s kubectl exec -it $HEAD_POD -- python Schola/Resources/python/schola/scripts/ray/launch.py \  
      --num-learners <num_learners> --num-cpus-per-learner <num_cpus_per_learner> \  
      --activation <activation_function> --launch-unreal \  
      --unreal-path "<path_to_unreal_executable>" -t <training_iterations> \  
      --save-final-policy -p <port> --headless <APPO/IMPALA>
  
   **Key Components of the Command**:  
  
   - ``--unreal-path "<path_to_unreal_executable>"``: This parameter specifies the path to a fully built Unreal Engine executable. It is crucial that this executable is included in the Docker image and accessible at runtime. The Unreal Engine instance is launched as part of the training process, enabling simulations or environments required for training.  
  
   - ``--num-learners <num_learners>``: Specifies the number of environment learners to use during training. This can be adjusted based on the available resources and the complexity of the task.  
  
   - ``--num-cpus-per-learner <num_cpus_per_learner>``: Defines the number of CPU cores allocated per learner. Adjust this parameter to optimize resource usage and performance.  
  
   - ``--activation <activation_function>``: Sets the activation function used in the neural network model. This can be modified to experiment with different activation functions.  
  
   - Additional parameters such as ``-t <training_iterations>`` (training iterations), ``--save-final-policy``, and ``-p <port>`` (port) can be customized to suit specific training requirements.  
  
   .. note::
      
      The options provided in this example are specific to a particular use case. There are many other options available to control the training process, and you should choose the ones that best fit your needs. Refer to the documentation of your training script for a comprehensive list of configurable parameters.  
  
#. **Monitor the Training Process**:  
  
   Optionally, start a TensorBoard instance to track the training:  
  
   .. code-block:: bash  
  
      #!/bin/bash  
  
      HEAD_POD=$(microk8s kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)  
      microk8s kubectl exec -it $HEAD_POD -- tensorboard --logdir /path/to/logs --host 0.0.0.0 