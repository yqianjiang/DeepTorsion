import numpy as np
import os
from skimage.transform import resize
import skvideo.io as skv
import matplotlib.pyplot as plt
from skimage import img_as_float
from skvideo.utils import rgb2gray
from utils.gen_data.seg_lib.Segmentation import getSegmentation_fromDL
from utils.gen_data.seg_lib.draw_ellipse import fit_ellipse
from .FindTorsion import genPolar, findTorsion
import logging


class torsion_inferer(object):
    def __init__(self, seg_model, torsion_model, my_logger):
        """
        Initialize necessary parameters and load deep_learning model

        Args:
            seg_model: Deep learning model that perform image segmentation.
            torsion_model: Deep model to infer torsional angle.
            infer_method: "hoi" -- interpolate during polar transform, then do cross-correlation (slow) 
                            "cc" -- interpolate after cross-correlation (fast)
                            "network" -- infer by deep network (simple network).
                            "stn" -- infer by DeepTorsion-STN.
        """
        self.seg_model = seg_model
        self.torsion_model = torsion_model
        self.logger = my_logger

    def predict(self, video_src, output_record, output_vis_path, infer_method, batch_size=32, print_prefix=""):
        """

        Parameters
        ----------
        video_src : str
            input video
        output_record : str
            record .csv
        batch_size : int
            Inference batch size
        print_prefix : str
            Printing identifier
        output_vis_path : str
            The path of the output visualization that will be drawn

        Returns
        -------

        """

        self.infer_method = infer_method
        if infer_method == "hoi":
            self.polar_resolution = 0.02
        elif infer_method == "cc":
            self.polar_resolution = 0.5
        elif infer_method == "network":
            self.polar_resolution = 0.5
        elif infer_method == "stn":
            self.polar_resolution = 0.5

        video_name_root, ext, vreader, (infervid_m, infervid_w, infervid_h,
                                        infervid_channels), shape_correct, image_scaling_factor = self._get_video_info(video_src)

        # Initialize path and video writer
        self.output_vid_path = output_vis_path
        self.vwriter = skv.FFmpegWriter(self.output_vid_path)

        # Initialize for result recording
        self.results_recorder = open(output_record, "w")
        self.results_recorder.write("frame, rotation\n")
        self.rotation_results = []
        self.time_display = 150  # range of frame when plotting graph

        final_batch_size = infervid_m % batch_size
        final_batch_idx = infervid_m - final_batch_size
        X_batch = np.zeros((batch_size, 240, 320, 1))
        X_batch_final = np.zeros((infervid_m % batch_size, 240, 320, 1))
        for idx, frame in enumerate(vreader.nextFrame()):
            print("\r%sInferring %s (%d%%)" % (print_prefix, video_name_root + ext, (idx / infervid_m) * 100), end="",
                  flush=True)
            frame_preprocessed = self._preprocess_image(frame, shape_correct)

            mini_batch_idx = idx % batch_size

            # Before reaching the batch size, stack the array
            if ((mini_batch_idx != 0) and (idx < final_batch_idx)) or (idx == 0):
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

            # After reaching the batch size, but not the final batch, predict and infer angles
            elif ((mini_batch_idx == 0) and (idx < final_batch_idx) or (idx == final_batch_idx)):
                Y_batch = self.seg_model.predict(X_batch)
                # =============== infer angles by batch here ====================
                self._infer_torsion_batch(X_batch, Y_batch, idx-batch_size)
                X_batch = np.zeros((batch_size, 240, 320, 1))
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

            # Within the final batch but not yet reaching the last index, stack the array
            elif ((idx > final_batch_idx) and (idx != infervid_m - 1)):
                X_batch_final[idx - final_batch_idx,
                              :, :, :] = frame_preprocessed

            # Within the final batch and reaching the last index, predict and infer angles
            elif (idx == infervid_m - 1):
                print("\r%sInferring %s (100%%)" %
                      (print_prefix, video_name_root + ext), end="", flush=True)
                X_batch_final[idx - final_batch_idx,
                              :, :, :] = frame_preprocessed
                Y_batch = self.seg_model.predict(X_batch_final)
                # =============== infer angles by batch here ====================
                self._infer_torsion_batch(X_batch_final, Y_batch, idx-batch_size)
            else:
                import pdb
                pdb.set_trace()
        self.results_recorder.close()
        self.vwriter.close()
        del self.template
        del self.r_template
        del self.theta_template
        del self.extra_radian

    def _infer_torsion_batch(self, X_batch, Y_batch, idx, update_template=False):
        # do visulisation for torsion
        refer_frame = 0
        for batch_idx, Y_each in enumerate(Y_batch):
            frame_id = idx + batch_idx
            pred = Y_each  # Y_each.shape (240, 320, 4)
            pred_masked = np.ma.masked_where(pred < 0.5, pred)

            # Initialize frames and maps
            frame = X_batch[batch_idx,:,:, 0] 
            frame_gray = frame   # frame_gray.shape (240, 320)
            frame_rgb = np.zeros((frame_gray.shape[0], frame_gray.shape[1], 3))
            frame_rgb[:, :, :] = frame_gray.reshape(
                frame_gray.shape[0], frame_gray.shape[1], 1)
            useful_map, (pupil_map, _, _, _) = getSegmentation_fromDL(pred)
            _, (pupil_map_masked, iris_map_masked, glints_map_masked,
                visible_map_masked) = getSegmentation_fromDL(pred_masked)
            rr, _, centre, _, _, _, _, _ = fit_ellipse(pupil_map, 0.5)

            # Cross-correlation
            if centre == None:
                refer_frame+=1
                self.logger.info("frame " + str(frame_id) + " : centre is None")
                continue
            elif frame_id == refer_frame:
                try:
                    self.template, __, self.r_template, self.theta_template, self.extra_radian = genPolar(frame_gray, useful_map, center=centre, template=True,
                                                                                                                               filter_sigma=100, adhist_times=2, resolution=self.polar_resolution)
                    rotated_info = (self.template,
                                    self.r_template, self.theta_template)
                    rotation = 0
                    self.logger.info("frame "+str(frame_id) +
                                     " is set as the first template")
                except:
                    import pdb
                    pdb.set_trace()
            elif rr is not None:                
                try:
                    rotation, rotated_info, _ = findTorsion(self.template, frame_gray, useful_map, center=centre, polar_resolution = self.polar_resolution,
                                                            filter_sigma=100, adhist_times=2, method=self.infer_method, model=self.torsion_model)

                except:
                    if frame_id < 2:
                        self.template, __, self.r_template, self.theta_template, self.extra_radian = genPolar(frame_gray, useful_map, center=centre, template=True,
                                                                                                                                filter_sigma=100, adhist_times=2, resolution=self.polar_resolution)
                        rotated_info = (self.template,
                                        self.r_template, self.theta_template)
                        rotation = 0
                        self.logger.info("frame "+str(frame_id)+" is set as template because error.")
                    else:
                        rotation, rotated_info, _ = findTorsion(self.template, frame_gray, useful_map, center=centre, polar_resolution = self.polar_resolution,
                                                            filter_sigma=100, adhist_times=2, method=self.infer_method, model=self.torsion_model)

                if (update_template == True) and rotation == 0:
                    self.template, __, self.r_template, self.theta_template, self.extra_radian = genPolar(frame_gray, useful_map, center=centre, template=True,
                                                                                                                               filter_sigma=100, adhist_times=2, resolution=self.polar_resolution)
            else:
                rotation, rotated_info = np.nan, None
                self.logger.info('rr is None, rotation set to np.nan')

            self.rotation_results.append(rotation)
            self.results_recorder.write(
                "{},{}\n".format(frame_id, rotation))

            # Drawing the frames of visualisation video
            rotation_plot_arr = self._plot_rotation_curve(frame_id)
            segmented_frame = self._draw_segmented_area(
                frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked)
            polar_transformed_graph_arr = self._plot_polar_transformed_graph(
                    (self.template, self.r_template, self.theta_template), rotated_info, self.extra_radian)
            frames_to_draw = (frame_rgb, rotation_plot_arr,
                              segmented_frame, polar_transformed_graph_arr)
            final_output = self._build_final_output_frame(frames_to_draw)
            self.vwriter.writeFrame(final_output)

    def _plot_rotation_curve(self, idx, y_lim=(-4, 4)):
        fig, ax = plt.subplots(figsize=(3.2, 2.4))  # width, height

        try:
            if idx < self.time_display:
                ax.plot(
                    np.arange(0, idx), self.rotation_results[0:idx], color="b", label="DeepVOG 3D")
                ax.set_xlim(0, self.time_display)
            else:
                ax.plot(np.arange(idx - self.time_display, idx),
                        self.rotation_results[idx-self.time_display:idx], color="b", label="DeepVOG 3D")
                ax.set_xlim(idx-self.time_display, idx)
            ax.legend()
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_yticks(np.arange(y_lim[0], y_lim[1]))
            plt.tight_layout()
        except:
            pass
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf

    def _draw_segmented_area(self, frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked):
        # Plot segmented area
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        ax.imshow(frame_gray, vmax=1, vmin=0, cmap="gray")
        ax.imshow(visible_map_masked, cmap="autumn", vmax=1, vmin=0, alpha=0.2)
        ax.imshow(iris_map_masked, cmap="GnBu", vmax=1, vmin=0, alpha=0.2)
        ax.imshow(pupil_map_masked, cmap="hot", vmax=1, vmin=0, alpha=0.2)
        ax.imshow(glints_map_masked, cmap="OrRd", vmax=1, vmin=0, alpha=0.2)
        ax.set_axis_off()
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf

    def _plot_polar_transformed_graph(self, template_info, rotated_info, extra_radian):
        (polar_pattern, r, theta) = template_info
        if rotated_info is not None:
            (polar_pattern_rotated, r_rotated, theta_rotated) = rotated_info
        else:
            polar_pattern_rotated, r_rotated, theta_rotated = np.zeros(
                polar_pattern.shape), r, theta

        # x axis correction
        theta_longer = np.rad2deg(
            theta) - np.rad2deg((theta.max()-theta.min())/2)
        theta_shorter = np.rad2deg(
            theta_rotated) - np.rad2deg((theta_rotated.max() - theta_rotated.min())/2)
        #theta_extra = np.rad2deg(extra_radian)
        

        # Plotting
        fig, ax = plt.subplots(2, figsize=(3.2, 2.4))
        ax[0].imshow(polar_pattern, cmap="gray", extent=(
            theta_shorter.min(), theta_shorter.max(), r.max(), r.min()), aspect='auto')
        ax[0].set_title("Template")
        ax[1].imshow(polar_pattern_rotated, cmap="gray", extent=(theta_shorter.min(
        ), theta_shorter.max(), r_rotated.max(), r_rotated.min()), aspect='auto')
        ax[1].set_title("Rotated pattern")
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf

    def _build_final_output_frame(self, frames_to_draw):
        """
        args:
            frames_to_draw: tuple with length 4. Starting from top left corner in clockwise direction.
        """
        height, width = 240, 320
        final_output = np.zeros((height*2, width*2, 3))
        final_output[0:height, 0:width, :] = frames_to_draw[0]
        final_output[0:height, width:width*2, :] = frames_to_draw[1]
        final_output[height:height*2, 0:width, :] = frames_to_draw[2]
        final_output[height:height*2, width:width*2, :] = frames_to_draw[3]
        final_output = (final_output*255).astype(np.uint8)
        return final_output

    def _get_video_info(self, video_src):
        video_name_with_ext = os.path.split(video_src)[1]
        video_name_root, ext = os.path.splitext(video_name_with_ext)
        vreader = skv.FFmpegReader(video_src)
        m, w, h, channels = vreader.getShape()
        print("get shape:", m, w, h, channels)
        image_scaling_factor = np.linalg.norm(
            (240, 320))/np.linalg.norm((h, w))
        shape_correct = self._inspectVideoShape(w, h)
        return video_name_root, ext, vreader, (m, w, h, channels), shape_correct, image_scaling_factor

    @staticmethod
    def _inspectVideoShape(w, h):
        if (w, h) == (240, 320):
            return True
        else:
            return False

    @staticmethod
    def _preprocess_image(img, shape_correct):
        """

        Args:
            img (numpy array): unprocessed image with shape (w, h, 3) and values int [0, 255]
        Returns:
            output_img (numpy array): processed grayscale image with shape ( 240, 320, 1) and values float [0,1]
        """
        output_img = np.zeros((240, 320, 1))
        img = img/255
        img = rgb2gray(img)
        if not shape_correct:
           img = resize(img, (240, 320, 1))
        output_img[:, :, :] = img.reshape(240, 320, 1)
        return output_img

    def __del__(self):
        pass
