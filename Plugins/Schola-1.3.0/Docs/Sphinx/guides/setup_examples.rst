Getting Started With Schola Examples
====================================

This guide will walk you through setting up the Schola Experiments Project to try out some premade reinforcement learning examples with Schola.


.. include:: ./doc_fragments/prerequisites.rst


Install Schola Examples
-----------------------

1. Get the `ScholaExamples` project, either by downloading the source, or via git.

   .. tabs::

      .. tab:: Manual Download 
            
         1. Download the zipped source code from the `Schola Examples Repository <https://github.com/GPUOpen-LibrariesAndSDKs/ScholaExamples>`__
         2. Unzip the project.

      .. tab:: Clone Repository

         .. code-block:: bash

            git clone --recurse-submodules https://github.com/GPUOpen-LibrariesAndSDKs/ScholaExamples.git

         .. note::
            
            If you experience an error installing the repository with git due to a large file sizes, run the following command to increase the git buffer size:
                
            .. code-block:: bash
                    
               git config -–global http.postBuffer 524288000

2. 
   .. include:: ./doc_fragments/pip_install.rst

3. Launch the Project.
   
.. note::

    For setting up your project files to run ScholaExample, please refer to the guide on `Project Files for IDEs <https://dev.epicgames.com/documentation/en-us/unreal-engine/how-to-generate-unreal-engine-project-files-for-your-ide>`_ from Epic Games.