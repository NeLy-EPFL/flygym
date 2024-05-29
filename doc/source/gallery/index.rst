Gallery
=======

NeuroMechFly can be used to emulate a wide range of behaviours and scenarios. Here are some examples of the experiments that can be conducted using Flygym.

.. raw:: html

   <style>
   .gallery-grid {
       display: grid;
       grid-template-columns: repeat(2, 1fr); /* Adjust number of columns as needed */
       /*row-gap: 0px;
       column-gap: 50px;
       width: 60%;*/
       gap: 20px;
    }
   .gallery-item {
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid grey;
        padding: 5px
    }
    .gallery-item p {
       margin-top: -5px;
       text-align: left;
       box-sizing: border-box;
       font-size: 15px;
   }
   .gallery-item:hover img {
        filter: grayscale(100%); /* Apply grayscale filter on hover */
    }
   </style>

   <div class="gallery-grid">
        <div class="gallery-item">
           <a href="video_3_forces.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_3_force_visualization_v7_TL_thumbnail.jpeg/?raw=true" alt="Force readout">
               <p>Replay walking behaviour and estimate contact forces</p>
           </a>
       </div>
       <div class="gallery-item">
           <a href="video_4_climbing.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_4_climbing_v8_TL_thumbnail.jpeg/?raw=true" alt="Climbing">
               <p>Walking on inclined terrain</p>
           </a>
       </div>
       <div class="gallery-item">
           <a href="video_8_controller_comparison.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_8_controller_comparison_v10_TL_small_thumbnail.jpeg/?raw=true" alt="Controller comparison">
               <p>Benchmarking various controllers on different terrains</p>
           </a>
       </div>
       <div class="gallery-item">
           <a href="video_9_visual_taxis.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_9_visual_taxis_no_stable_v14_TL_thumbnail.jpeg/?raw=true" alt="Visual taxis">
               <p>Emulating vision and abstract visual processing</p>
           </a>
       </div>
       <div class="gallery-item">
           <a href="video_10_odour_taxis.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_10_odor_taxis_v8_TL_thumbnail.jpeg/?raw=true" alt="Odour taxis">
               <p>Emulating simple olfaction and abstract olfactory processing</p>
           </a>
        </div>
       <div class="gallery-item">
           <a href="video_11_head_stabilization.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_11_head_stabilization_thumbnail.jpeg/?raw=true" alt="Head stabilization">
               <p>Using ascending sensory information to stabilize the head on complex terrain</p>
           </a>
       </div>
       <div class="gallery-item">
           <a href="video_12_multimodal_navigation.html">
               <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_12_multimodal_navigation_example_v3_TL_thumbnail.jpeg/?raw=true" alt="Multimodal navigation">
               <p>Building reinforcement learning trained controllers to solve multimodal tasks.</p>
           </a>
       </div>
        <div class="gallery-item">
            <a href="video_13_plume_navigation.html">
                <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_13_plume_navigation_v2_SWC_thumbnail.jpeg/?raw=true" alt="Plume navigation">
                <p>Navigating dynamic olfactive environements</p>
            </a>
        </div>
        <div class="gallery-item">
            <a href="video_14_fly_follow_fly.html">
                <img src="https://github.com/NeLy-EPFL/_media/blob/main/flygym/videos/video_14_fly_follow_fly_v6_SWC_thumbnail.jpeg?raw=true" alt="Fly follow fly">
                <p>Building complex social scenarios relying on complete sensory-motor loops with biorealistic sensing. </p>
            </a>
        </div>
   </div>
