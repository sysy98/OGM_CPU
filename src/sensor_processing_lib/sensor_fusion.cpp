/******************************************************************************
 *
 * Author: Shuai Yuan
 * Date: 01/07/2021
 *
 */

#include <sensor_processing_lib/sensor_fusion.h>
#include <math.h>
#include <algorithm>
#include <chrono>


namespace sensor_processing
{

	ros::Time SensorFusion::cloud_stamp_{};                 // cloud stamp
	int SensorFusion::frame_count_{};                        // init counter for publishing
	Matrix4f SensorFusion::transMat = Matrix4f::Identity(); // init transformation matrix

	// Initialize process duration
	std::chrono::microseconds total_dur{};
	std::chrono::microseconds main_dur{};
	/******************************************************************************/
	SensorFusion::SensorFusion(ros::NodeHandle nh, ros::NodeHandle private_nh) : nh_(nh),
																				 private_nh_(private_nh),
																				 pcl_in_(new VPointCloud),
																				 pcl_ground_plane_(new VPointCloud),
																				 pcl_elevated_(new VPointCloud),
																				 cloud_sub_(nh, "/kitti/velo/pointcloud", 10),
																				 tf_listener_(buffer_),
																				 width_gain_(2),
																				 height_gain_(1.5)
	{

		// Get data path
		std::string home_dir;
		if (ros::param::get("~home_dir", home_dir))
		{
			params_.home_dir = home_dir + "/master_ws/kitti_data";
		}
		else
		{
			ROS_ERROR("Set dataset path as parameter");
		}

		// Get scenario parameter
		int scenario;
		if (ros::param::get("~scenario", scenario))
		{
			std::ostringstream scenario_stream;
			scenario_stream << std::setfill('0') << std::setw(4) << scenario;
			params_.scenario = scenario_stream.str();
		}
		else
		{
			ROS_ERROR("Failed to read scenario");
		}

		// Define lidar parameters
		private_nh_.param("lidar/height", params_.lidar_height,
						  params_.lidar_height);
		private_nh_.param("lidar/z_min", params_.lidar_z_min,
						  params_.lidar_z_min);

		// Define grid parameters
		private_nh_.param("grid/range/min", params_.grid_range_min,
						  params_.grid_range_min);
		private_nh_.param("grid/range/max", params_.grid_range_max,
						  params_.grid_range_max);
		private_nh_.param("grid/cell/size", params_.grid_cell_size,
						  params_.grid_cell_size);
		private_nh_.param("grid/cell/height", params_.grid_cell_height,
						  params_.grid_cell_height);
		private_nh_.param("grid/segments", params_.grid_segments,
						  params_.grid_segments);

		params_.grid_bins = (params_.grid_range_max * std::sqrt(2)) /
								params_.grid_cell_size + 1;
		params_.polar_grid_num = params_.grid_segments * params_.grid_bins;

		// 360-degree grid map
		params_.height_grid = params_.grid_range_max / params_.grid_cell_size * 2;
		params_.width_grid = params_.height_grid;

		// Define ransac ground plane parameters
		private_nh_.param("ransac/tolerance", params_.ransac_tolerance,
						  params_.ransac_tolerance);
		private_nh_.param("ransac/iterations", params_.ransac_iterations,
						  params_.ransac_iterations);

		// Define static conversion values
		params_.inv_angular_res = params_.grid_segments / (2 * M_PI);
		params_.inv_radial_res = 1.0f / params_.grid_cell_size;

		// Print parameters
		ROS_INFO_STREAM("scenario " << params_.scenario);
		ROS_INFO_STREAM("lidar_height " << params_.lidar_height);
		ROS_INFO_STREAM("lidar_z_min " << params_.lidar_z_min);
		ROS_INFO_STREAM("grid_range_min " << params_.grid_range_min);
		ROS_INFO_STREAM("grid_range_max " << params_.grid_range_max);
		ROS_INFO_STREAM("height_grid " << params_.height_grid);
		ROS_INFO_STREAM("width_grid " << params_.width_grid);
		ROS_INFO_STREAM("grid_cell_size " << params_.grid_cell_size);
		ROS_INFO_STREAM("grid_cell_height " << params_.grid_cell_height);
		ROS_INFO_STREAM("grid_bins " << params_.grid_bins);
		ROS_INFO_STREAM("grid_segments " << params_.grid_segments);
		ROS_INFO_STREAM("ransac_tolerance " << params_.ransac_tolerance);
		ROS_INFO_STREAM("ransac_iterations " << params_.ransac_iterations);
		ROS_INFO_STREAM("inv_angular_res " << params_.inv_angular_res);
		ROS_INFO_STREAM("inv_radial_res " << params_.inv_radial_res);

		// Define polar grid
		polar_grid_ = std::vector<PolarCell>(params_.polar_grid_num);

		// final OGM
		params_.occ_width_grid = width_gain_ * params_.width_grid;
		params_.occ_height_grid = height_gain_ * params_.height_grid;
		occ_grid_ = boost::make_shared<OccupancyGrid>();
		occ_grid_->data.resize(params_.occ_width_grid * params_.occ_height_grid);
		occ_grid_->info.width = uint32_t(params_.occ_height_grid);
		occ_grid_->info.height = uint32_t(params_.occ_width_grid);
		occ_grid_->info.resolution = float(params_.grid_cell_size);
		occ_grid_->info.origin.position.x = -params_.occ_height_grid / 2 * params_.grid_cell_size;
		occ_grid_->info.origin.position.y = -params_.occ_width_grid / 2 * params_.grid_cell_size;
		occ_grid_->info.origin.position.z = params_.lidar_height;
		occ_grid_->info.origin.orientation.w = 1;
		occ_grid_->info.origin.orientation.x = 0;
		occ_grid_->info.origin.orientation.y = 0;
		occ_grid_->info.origin.orientation.z = 0;

		whole_grid_probs_ = std::vector<float>(params_.occ_height_grid * params_.occ_width_grid, 0.0);

		// Init occupancy grid
		for (int i = 0; i < params_.occ_width_grid * params_.occ_height_grid; ++i)
		{
			occ_grid_->data[i] = -1;
		}

		// Define Publisher
		grid_occupancy_pub_ = nh_.advertise<OccupancyGrid>(
			"/sensor/grid/occupancy", 2);
		vehicle_pos_pub_ = nh_.advertise<PointStamped>(
			"/vehicle_pose", 2);

		// Define Subscriber
		cloud_sub_.registerCallback(boost::bind(&SensorFusion::process, this, _1));
	}

	SensorFusion::~SensorFusion() {}

	void SensorFusion::process(
		const PointCloud2::ConstPtr &cloud)
	{
		auto startPost = std::chrono::high_resolution_clock::now();
		cloud_stamp_ = cloud->header.stamp;

		calculateTransMatrix();

		// Preprocess point cloud
		processPointCloud(cloud);

	    auto endPost = std::chrono::high_resolution_clock::now();

		// sum the total time
    	total_dur += std::chrono::duration_cast<std::chrono::microseconds>(endPost - startPost);

		std::cout << "Average time (whole process): " << std::fixed << std::setprecision(2) << double(total_dur.count()) / (frame_count_ + 1) / 1000 << "ms"
				<< "(main process part): " << std::fixed << std::setprecision(2) << double(main_dur.count()) / (frame_count_ + 1) / 1000 << "ms" << std::endl;
		
		// Increment time frame
		frame_count_++;
	}

	void SensorFusion::processPointCloud(const PointCloud2::ConstPtr &cloud)
	{

		/******************************************************************************
 * 1. Filter point cloud to only consider points in the front that can also be
 * found in image space.
 */

		// Convert input cloud
		pcl::fromROSMsg(*cloud, *pcl_in_);

		// Define point_cloud_inliers and indices
		pcl::PointIndices::Ptr pcl_inliers(new pcl::PointIndices());
		pcl::ExtractIndices<VPoint> pcl_extractor;

		// Define polar grid
		polar_grid_ = std::vector<PolarCell>(params_.polar_grid_num, PolarCell());

		// Loop through input point cloud
		for (int i = 0; i < pcl_in_->size(); ++i)
		{

			// Read current point
			VPoint &point = pcl_in_->at(i);

			// Determine range of lidar point and check
			float range = std::sqrt(point.x * point.x + point.y * point.y);
			if (range > params_.grid_range_min &&
				range < params_.grid_range_max)
			{

				// Check height of lidar point
				if (point.z > params_.lidar_z_min)
				{

					// Add index for filtered point cloud
					pcl_inliers->indices.push_back(i);

					// Buffer variables
					int seg, polar_id;
					float mag;

					// Get polar grid cell indices
					fromVeloCoordsToPolarCell(point.x, point.y, seg, polar_id, mag);

					// Grab cell
					PolarCell &cell = polar_grid_[polar_id];

					// Increase count
					cell.count++;

					// Update min max
					if (cell.count == 1)
					{
						cell.x_min = point.x;
						cell.y_min = point.y;
						cell.z_min = point.z;
						cell.z_max = point.z;
					}
					else
					{
						if (point.z < cell.z_min)
						{
							cell.x_min = point.x;
							cell.y_min = point.y;
							cell.z_min = point.z;
						}
						if (point.z > cell.z_max)
						{
							cell.z_max = point.z;
						}
					}
				}
			}
		}

		// Extract points from original point cloud
		pcl_extractor.setInputCloud(pcl_in_);
		pcl_extractor.setIndices(pcl_inliers);
		pcl_extractor.setNegative(false);
		pcl_extractor.filter(*pcl_in_);

		// Publish filtered cloud
		pcl_in_->header.frame_id = cloud->header.frame_id;
		pcl_in_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);

		/******************************************************************************
 * 2. Ground plane estimation and dividing point cloud in elevated and ground
 */
		// Clear ground plane points
		pcl_ground_plane_->points.clear();

		// Loop over cartesian grid map
		for (int i = 0; i < params_.grid_segments; ++i)
		{
			for (int j = 0; j < params_.grid_bins; ++j)
			{

				// Grab cell
				int polar_id = from2dPolarIndexTo1d(i, j);
				PolarCell &cell = polar_grid_[polar_id];

				// Check if cell can be ground cell
				if (cell.count > 0 &&
					(cell.z_max - cell.z_min < params_.grid_cell_height))
				{

					// Push back cell attributes to ground plane cloud
					pcl_ground_plane_->points.push_back(
						VPoint(cell.x_min, cell.y_min, cell.z_min));
				}
			}
		}

		// Estimate the ground plane using PCL and RANSAC
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

		// Create the segmentation object
		pcl::SACSegmentation<VPoint> segmentation;
		segmentation.setOptimizeCoefficients(true);
		segmentation.setModelType(pcl::SACMODEL_PLANE);
		segmentation.setMethodType(pcl::SAC_RANSAC);
		segmentation.setDistanceThreshold(params_.ransac_tolerance);
		segmentation.setMaxIterations(params_.ransac_iterations);
		segmentation.setInputCloud(pcl_ground_plane_->makeShared());
		segmentation.segment(*inliers, *coefficients);

		// Sanity check
		if (inliers->indices.empty() || coefficients->values[3] > 2 ||
			coefficients->values[3] < 1.5)
		{
			ROS_WARN("Bad ground plane estimation! # Ransac Inliers [%d] # Lidar "
					 "height [%f]",
					 int(inliers->indices.size()),
					 coefficients->values[3]);
		}

		/******************************************************************************
 * 3. Divide filtered point cloud in elevated and ground
 */
		for (int s = 0; s < params_.grid_segments; ++s)
		{

			for (int b = 0; b < params_.grid_bins; ++b)
			{

				// Grab cell
				int polar_id = from2dPolarIndexTo1d(s, b);
				PolarCell &cell = polar_grid_[polar_id];
				float x, y;

				//Calcluate distance between the cell and the origin of lidar
				cell.dist = b / params_.inv_radial_res + params_.grid_cell_size / 2;

				fromPolarCellToVeloCoords(s, b, x, y);

				//Get ground height
				cell.ground = (-coefficients->values[0] * x -
							   coefficients->values[1] * y - coefficients->values[3]) /
							   coefficients->values[2];

				//if cell is not filled
				if (cell.count == 0)
					continue;

				//Calculate cell height
				else
					cell.height = cell.z_max - cell.ground;
			}
		}

		pcl_elevated_->points.clear();

		for (int i = 0; i < pcl_in_->size(); ++i)
		{

			// Read current point
			VPoint point = pcl_in_->at(i);

			//Buffer variables
			int seg, polar_id;
			float mag;

			// Get polar grid cell indices
			fromVeloCoordsToPolarCell(point.x, point.y, seg, polar_id, mag);

			// Grab cell
			PolarCell &cell = polar_grid_[polar_id];

			if (point.z > cell.ground && cell.height > params_.grid_cell_height)
			{
				pcl_elevated_->points.push_back(point);
			}
		}

		pcl_elevated_->header.frame_id = cloud->header.frame_id;
		pcl_elevated_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);

		// Print point cloud information
    	ROS_INFO("Point Cloud [%d] # Total points [%d] # Elevated points [%d] ",
             frame_count_, int(pcl_in_->size()), int(pcl_elevated_->size()));

		/******************************************************************************
 * 4. Use elevated point cloud to calculate occupied probability of cells in polar grid
 */
		clock_t startTime,endTime;
		startTime = clock();

		for (int i = 0; i < pcl_elevated_->size(); ++i)
		{

			// Read current point
			VPoint &point = pcl_elevated_->at(i);

			//Buffer variables
			int seg, polar_id;
			float point_dist;

			fromVeloCoordsToPolarCell(point.x, point.y, seg, polar_id, point_dist);

			// if(i==0) printf("%d* ", seg);
			PolarCell &cell = polar_grid_[polar_id];
			cell.elevated_count++;
		
			// Calculate p_occ with the normal distribuiton.
			for (int j = 0; j < params_.grid_bins; ++j)
			{
				int tmp_polar_id = from2dPolarIndexTo1d(seg, j);
				PolarCell &cell = polar_grid_[tmp_polar_id];

				// float alpha = log10 (500 / point_dist);
				float alpha = 1;
				float delta = 0.25;
				float occ_value = alpha / sqrt(2 * M_PI * delta) *
							  exp(-pow(cell.dist - point_dist, 2) / (2 * pow(delta, 2)));
				cell.p_occ += occ_value;
			}
		}

		for (int i = 0; i < params_.grid_segments; ++i)
		{

			bool has_obstacle_this_seg = false;
			int first_occ_cell_bin{-1};

			for (int j = 0; j < params_.grid_bins; ++j)
			{
				int polar_id = from2dPolarIndexTo1d(i, j);
				PolarCell &cell = polar_grid_[polar_id];

				// Perform nonlinear normalization of p_occ in the range of 0-1.
				cell.p_occ = std::atan(cell.p_occ) * 2 / M_PI;

				if (cell.elevated_count && !has_obstacle_this_seg)
				{
					first_occ_cell_bin = j;
					has_obstacle_this_seg = true;
				}

				// p_free
				if (cell.dist < params_.grid_range_min)
				{
					cell.p_free = 0.4 * params_.grid_range_min / params_.grid_range_max;
				}
				else if (cell.dist <= params_.grid_range_max)
				{
					cell.p_free = 0.4 * cell.dist / params_.grid_range_max;
				}
			}

			if (!has_obstacle_this_seg)
				continue;

			for (int j = 0; j < params_.grid_bins; ++j)
			{

				int polar_id = from2dPolarIndexTo1d(i, j);
				PolarCell &cell = polar_grid_[polar_id];

				//The probability of the cell behind the first detected object is unknown(0.5).
				if (j > first_occ_cell_bin)
				{
					cell.p_free = 0.5;
				}

				cell.p_final = max(cell.p_occ, cell.p_free);
				cell.p_logit = log(cell.p_final / (1 - cell.p_final));
			}
		}
		/******************************************************************************
 * 5. Map polar grid back to cartesian occupancy grid
 */		
		auto main_part_start = std::chrono::high_resolution_clock::now();

		// Go through cartesian grid(occ_grid_)
		float x1 = -params_.grid_range_max + params_.grid_cell_size / 2;
		for (int i = 0; i < params_.height_grid; ++i, x1 += params_.grid_cell_size)
		{

			float y1 = -params_.grid_range_max + params_.grid_cell_size / 2;
			for (int j = 0; j < params_.width_grid; ++j, y1 += params_.grid_cell_size)
			{

				//Buffer variables
				int seg, polar_id;
				float mag;

				//Get polar grid indices
				fromVeloCoordsToPolarCell(x1, y1, seg, polar_id, mag);

				PolarCell &cell = polar_grid_[polar_id];

				// Calculate occupancy grid cell index
				int final_grid_index;

				fromLocalOgmToFinalOgm(i, j, final_grid_index);

				whole_grid_probs_[final_grid_index] += cell.p_logit;

				float final_prob = whole_grid_probs_[final_grid_index];

				if (final_prob == 0)
				{
					occ_grid_->data[final_grid_index] = -1;
				}
				else if (final_prob < 0)
				{
					occ_grid_->data[final_grid_index] = 0;
				}
				else
				{
					occ_grid_->data[final_grid_index] = 100;
				}
			}
		}
		
		auto main_part_end = std::chrono::high_resolution_clock::now();

		// Sum the processing time for the main part
		main_dur += std::chrono::duration_cast<std::chrono::microseconds>(main_part_end - main_part_start);

		// Publish occupancy grid
		occ_grid_->header.stamp = cloud->header.stamp;
		occ_grid_->header.frame_id = "world";
		occ_grid_->info.map_load_time = cloud->header.stamp;
		grid_occupancy_pub_.publish(occ_grid_);

		// Publish vehicle pose
		Vector4f vehicle_vec = transMat * Vector4f(0, 0, 0, 1);
		vehicle_pos_.header.frame_id = "world";
		vehicle_pos_.header.stamp = cloud_stamp_;
		vehicle_pos_.point.x = vehicle_vec[0];
		vehicle_pos_.point.y = vehicle_vec[1];
		vehicle_pos_.point.z = 0;
		vehicle_pos_pub_.publish(vehicle_pos_);	
	}

	void SensorFusion::calculateTransMatrix()
	{
		geometry_msgs::TransformStamped tfStamped;

		try
		{
			tfStamped = buffer_.lookupTransform("world", "velo_link", cloud_stamp_, ros::Duration(1.0));
		}
		catch (tf2::TransformException &ex)
		{
			ROS_WARN("%s", ex.what());
			ros::Duration(1.0).sleep();
		}
		//translation vector T
		auto trans = tfStamped.transform.translation;
		transMat.block<3, 1>(0, 3) = Vector3f(trans.x, trans.y, trans.z);

		//rotation matrix R
		auto rot = tfStamped.transform.rotation;
		transMat.block<3, 3>(0, 0) = Quaternionf(rot.w, rot.x, rot.y, rot.z)
										 .normalized()
										 .toRotationMatrix();
	}

	void SensorFusion::fromLocalOgmToFinalOgm(const int local_grid_x,
											  const int local_grid_y, int &final_grid_index)
	{

		float final_x, final_y;
		fromLocalGridToGlobalCartesian(local_grid_x, local_grid_y, final_x, final_y);
		fromFinalCartesianToGridIndex(final_x, final_y, final_grid_index);
	}

	void SensorFusion::fromLocalGridToGlobalCartesian(const int grid_x, const int grid_y,
													 float &global_x, float &global_y)
	{
		float x = (grid_x + 0.5 - params_.height_grid / 2) * params_.grid_cell_size;
		float y = (grid_y + 0.5 - params_.width_grid / 2) * params_.grid_cell_size;
		Vector4f new_vec = transMat * Vector4f(x, y, 0, 1);
		global_x = new_vec[0];
		global_y = new_vec[1];
	}

	void SensorFusion::fromFinalCartesianToGridIndex(const float x, const float y,
													 int &grid_index)
	{

		int grid_x = x / params_.grid_cell_size + params_.occ_height_grid / 2;
		int grid_y = y / params_.grid_cell_size + params_.occ_width_grid / 2;

		if (grid_x >= 0 && grid_x < params_.occ_height_grid && grid_y >= 0 && grid_y < params_.occ_width_grid)
		{

			grid_index = grid_y * params_.occ_height_grid + grid_x;
		}
	}

	inline int SensorFusion::from2dPolarIndexTo1d(const int seg, const int bin)
	{
		return seg * params_.grid_bins + bin;
	}

	void SensorFusion::fromVeloCoordsToPolarCell(const float x, const float y,
												 int &seg, int &polar_id, float &mag)
	{

		float ang = std::atan2(x, y);
		mag = std::sqrt(x * x + y * y);
		seg = int((ang + M_PI) * params_.inv_angular_res);
		int bin = int(mag * params_.inv_radial_res);
		polar_id = from2dPolarIndexTo1d(seg, bin);
	}

	void SensorFusion::fromPolarCellToVeloCoords(const int seg, const int bin,
												 float &x, float &y)
	{

		float mag = bin / params_.inv_radial_res + params_.grid_cell_size / 2;
		float ang = seg / params_.inv_angular_res - M_PI;
		x = std::sin(ang) * mag;
    	y = std::cos(ang) * mag;
	}
} // namespace sensor_processing
