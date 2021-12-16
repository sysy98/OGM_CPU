/******************************************************************************
 *
 * Author: Shuai Yuan
 * Date: 01/07/2021
 *
 */

#include <sensor_processing_lib/sensor_fusion.h>
#include <math.h>
#include <algorithm>
#include <time.h>


namespace sensor_processing
{

	ros::Time SensorFusion::cloud_stamp_{};                 // cloud stamp
	int SensorFusion::time_frame_{};                        // init counter for publishing
	Matrix4f SensorFusion::transMat = Matrix4f::Identity(); // init transformation matrix

	/******************************************************************************/
	SensorFusion::SensorFusion(ros::NodeHandle nh, ros::NodeHandle private_nh) : nh_(nh),
																				 private_nh_(private_nh),
																				 pcl_in_(new VPointCloud),
																				 pcl_ground_plane_(new VPointCloud),
																				 pcl_ground_plane_inliers_(new VPointCloud),
																				 pcl_ground_plane_outliers_(new VPointCloud),
																				 pcl_ground_(new VPointCloud),
																				 pcl_elevated_(new VPointCloud),
																				 cloud_sub_(nh, "/kitti/velo/pointcloud", 10),
																				 tf_listener_(buffer_),
																				 width_gain_(2),
																				 height_gain_(3.5)
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
		params_.lidar_opening_angle = M_PI / 2;

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
		params_.height_grid = params_.grid_range_max / params_.grid_cell_size;
		params_.width_grid = params_.height_grid * 2;
		params_.grid_bins = (params_.grid_range_max * std::sqrt(2)) /
								params_.grid_cell_size +
							1;

		// Define ransac ground plane parameters
		private_nh_.param("ransac/tolerance", params_.ransac_tolerance,
						  params_.ransac_tolerance);
		private_nh_.param("ransac/iterations", params_.ransac_iterations,
						  params_.ransac_iterations);

		// Define static conversion values
		params_.inv_angular_res = params_.grid_segments / (2 * params_.lidar_opening_angle);
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
		polar_grid_ = std::vector<std::vector<PolarCell>>(params_.grid_segments,
														  std::vector<PolarCell>(params_.grid_bins));

		// tmp OGM
		tmp_grid_ = boost::make_shared<OccupancyGrid>();
		tmp_grid_->data.resize(params_.width_grid * params_.height_grid);
		tmp_grid_->info.width = uint32_t(params_.height_grid);
		tmp_grid_->info.height = uint32_t(params_.width_grid);
		tmp_grid_->info.resolution = float(params_.grid_cell_size);
		tmp_grid_->info.origin.position.x = 0;
		tmp_grid_->info.origin.position.y = -params_.grid_range_max;
		tmp_grid_->info.origin.position.z = params_.lidar_height;
		tmp_grid_->info.origin.orientation.w = 1;
		tmp_grid_->info.origin.orientation.x = 0;
		tmp_grid_->info.origin.orientation.y = 0;
		tmp_grid_->info.origin.orientation.z = 0;

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
		for (int i = 0; i < params_.height_grid * params_.width_grid; ++i)
		{
			tmp_grid_->data[i] = -1;
		}
		for (int i = 0; i < params_.occ_width_grid * params_.occ_height_grid; ++i)
		{
			occ_grid_->data[i] = -1;
		}

		// Define Publisher
		cloud_filtered_pub_ = nh_.advertise<PointCloud2>(
			"/sensor/cloud/filtered", 2);
		cloud_ground_plane_inliers_pub_ = nh_.advertise<PointCloud2>(
			"/sensor/cloud/groundplane/inliers", 2);
		cloud_ground_plane_outliers_pub_ = nh_.advertise<PointCloud2>(
			"/sensor/cloud/groundplane/outliers", 2);
		cloud_ground_pub_ = nh_.advertise<PointCloud2>(
			"/sensor/cloud/ground", 2);
		cloud_elevated_pub_ = nh_.advertise<PointCloud2>(
			"/sensor/cloud/elevated", 2);
		tmp_occupancy_pub_ = nh_.advertise<OccupancyGrid>(
			"/sensor/grid/tmp_occupancy", 2);
		grid_occupancy_pub_ = nh_.advertise<OccupancyGrid>(
			"/sensor/grid/occupancy", 2);
		point_pub_ = nh_.advertise<PointStamped>(
			"/new_point", 2);
		fix_point_pub_ = nh_.advertise<PointStamped>(
			"/fixed_point", 2);

		// Define Subscriber
		cloud_sub_.registerCallback(boost::bind(&SensorFusion::process, this, _1));
	}

	SensorFusion::~SensorFusion() {}

	void SensorFusion::process(
		const PointCloud2::ConstPtr &cloud)
	{

		cloud_stamp_ = cloud->header.stamp;
		calculateTransMatrix();

		// Preprocess point cloud
		processPointCloud(cloud);

		// Print sensor fusion
		ROS_INFO("Publishing Sensor Fusion [%d]: # PCL points [%d] # Ground [%d]"
				 " # Elevated [%d] ",
				 time_frame_,
				 int(pcl_in_->size()), int(pcl_ground_->size()),
				 int(pcl_elevated_->size()));

		// Increment time frame
		time_frame_++;
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
		polar_grid_ = std::vector<std::vector<PolarCell>>(params_.grid_segments,
														  std::vector<PolarCell>(params_.grid_bins));

		// Loop through input point cloud
		for (int i = 0; i < pcl_in_->size(); ++i)
		{

			// Read current point
			VPoint &point = pcl_in_->at(i);

			// Determine angle of lidar point and check
			float angle = std::abs(std::atan2(point.y, point.x));
			if (angle < params_.lidar_opening_angle)
			{

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
						int seg, bin;

						// Get polar grid cell indices
						fromVeloCoordsToPolarCell(point.x, point.y, seg, bin);

						// Grab cell
						PolarCell &cell = polar_grid_[seg][bin];

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
		}

		// Extract points from original point cloud
		pcl_extractor.setInputCloud(pcl_in_);
		pcl_extractor.setIndices(pcl_inliers);
		pcl_extractor.setNegative(false);
		pcl_extractor.filter(*pcl_in_);

		// Publish filtered cloud
		pcl_in_->header.frame_id = cloud->header.frame_id;
		pcl_in_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
		cloud_filtered_pub_.publish(pcl_in_);

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
				PolarCell &cell = polar_grid_[i][j];

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

		// Divide ground plane cloud in inlier cloud and outlier cloud
		pcl_extractor.setInputCloud(pcl_ground_plane_);
		pcl_extractor.setIndices(inliers);
		pcl_extractor.setNegative(false);
		pcl_extractor.filter(*pcl_ground_plane_inliers_);

		pcl_extractor.setInputCloud(pcl_ground_plane_);
		pcl_extractor.setIndices(inliers);
		pcl_extractor.setNegative(true);
		pcl_extractor.filter(*pcl_ground_plane_outliers_);

		// Sanity check
		if (inliers->indices.empty() || coefficients->values[3] > 2 ||
			coefficients->values[3] < 1.5)
		{
			ROS_WARN("Bad ground plane estimation! # Ransac Inliers [%d] # Lidar "
					 "height [%f]",
					 int(inliers->indices.size()),
					 coefficients->values[3]);
		}

		// Publish ground plane inliers and outliers point cloud
		pcl_ground_plane_inliers_->header.frame_id = cloud->header.frame_id;
		pcl_ground_plane_inliers_->header.stamp =
			pcl_conversions::toPCL(cloud->header.stamp);
		cloud_ground_plane_inliers_pub_.publish(pcl_ground_plane_inliers_);

		pcl_ground_plane_outliers_->header.frame_id = cloud->header.frame_id;
		pcl_ground_plane_outliers_->header.stamp =
			pcl_conversions::toPCL(cloud->header.stamp);
		cloud_ground_plane_outliers_pub_.publish(pcl_ground_plane_outliers_);

		// Print
		ROS_INFO("Ground plane estimation [%d] # Points [%d] # Inliers [%d] "
				 " C [%f][%f][%f][%f]",
				 time_frame_,
				 int(pcl_ground_plane_->size()), int(pcl_ground_plane_inliers_->size()),
				 coefficients->values[0], coefficients->values[1],
				 coefficients->values[2], coefficients->values[3]);

		/******************************************************************************
 * 3. Divide filtered point cloud in elevated and ground
 */
		for (int s = 0; s < params_.grid_segments; ++s)
		{

			for (int b = 0; b < params_.grid_bins; ++b)
			{

				// Grab cell
				PolarCell &cell = polar_grid_[s][b];
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
					cell.height = polar_grid_[s][b].z_max - cell.ground;
			}
		}

		pcl_ground_->points.clear();
		pcl_elevated_->points.clear();

		for (int i = 0; i < pcl_in_->size(); ++i)
		{

			// Read current point
			VPoint point = pcl_in_->at(i);

			//Buffer variables
			int seg, bin;

			// Get polar grid cell indices
			fromVeloCoordsToPolarCell(point.x, point.y, seg, bin);

			// Grab cell
			PolarCell &cell = polar_grid_[seg][bin];

			if (point.z > cell.ground && cell.height > params_.grid_cell_height)
			{
				pcl_elevated_->points.push_back(point);
			}
			else
			{
				pcl_ground_->points.push_back(point);
			}
		}

		//Publish ground cloud
		pcl_ground_->header.frame_id = cloud->header.frame_id;
		pcl_ground_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
		cloud_ground_pub_.publish(pcl_ground_);

		//Publish elevated cloud
		pcl_elevated_->header.frame_id = cloud->header.frame_id;
		pcl_elevated_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
		cloud_elevated_pub_.publish(pcl_elevated_);

		/******************************************************************************
 * 4. Use elevated point cloud to calculate occupied probability of cells in polar grid
 */
		clock_t startTime,endTime;
		startTime = clock();

		int time{0};

		for (int i = 0; i < pcl_elevated_->size(); ++i)
		{

			// Read current point
			VPoint &point = pcl_elevated_->at(i);

			//Buffer variables
			int seg, bin;
			float point_dist;

			fromVeloCoordsToPolarInfo(point.x, point.y, seg, bin, point_dist);

			// if(i==0) printf("%d* ", seg);
			PolarCell &cell = polar_grid_[seg][bin];
			cell.elevated_count++;
		
			// Calculate p_occ with the normal distribuiton.
			for (int j = 0; j < params_.grid_bins; ++j)
			{

				PolarCell &cell = polar_grid_[seg][j];

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

				PolarCell &cell = polar_grid_[i][j];

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

			// Calculate the position of the first detected object in each segment.
			PolarCell &cell = polar_grid_[i][first_occ_cell_bin];

			cell.p_free = 0.5 + cell.elevated_count * 0.05;

			for (int j = 0; j < params_.grid_bins; ++j)
			{

				PolarCell &cell = polar_grid_[i][j];

				//The probability of the cell behind the first detected object is unknown(0.5).
				if (j > first_occ_cell_bin)
				{
					cell.p_free = 0.5;
				}

				cell.p_final = max(cell.p_occ, cell.p_free);
				cell.p_logit = log(cell.p_final / (1 - cell.p_final));
			}
		}

		// for (int i = 0; i < params_.grid_segments; ++i){
		// 	for (int j = 0; j < params_.grid_bins; ++j){
				
		// 		if(i == 25 && 10 < j < 15){
		// 			PolarCell &cell = polar_grid_[i][j];

		// 			printf("polar_id:%d, occ:%f, free:%f, final:%f, logit:%f \n  ", 
		// 			i * params_.grid_bins + j, cell.p_occ, cell.p_free, cell.p_final, cell.p_logit);
		// 		}
		// 	}
		// }
		/******************************************************************************
 * 5. Map polar grid back to cartesian occupancy grid
 */
		fixedPoint(0, 0);
		// Go through cartesian grid(occ_grid_)
		float x1 = params_.grid_cell_size / 2;
		for (int i = 0; i < params_.height_grid; ++i, x1 += params_.grid_cell_size)
		{

			float y1 = -params_.grid_range_max + params_.grid_cell_size / 2;
			for (int j = 0; j < params_.width_grid; ++j, y1 += params_.grid_cell_size)
			{

				//Buffer variables
				int seg, bin;

				//Get polar grid indices
				fromVeloCoordsToPolarCell(x1, y1, seg, bin);

				PolarCell &cell = polar_grid_[seg][bin];

				// Calculate occupancy grid cell index
				int final_grid_index;

				fromLocalOgmToFinalOgm(i, j, final_grid_index);

				if(j == 0 && i == 0)
					printf("velo_x: %f, velo_y: %f, seg:%d, polar_id:%d, final_id:%d \n  ",x1, y1, seg, seg * params_.grid_bins + bin, final_grid_index);

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
		endTime = clock();
		std::cout << "Total Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

		// Publish occupancy grid
		occ_grid_->header.stamp = cloud->header.stamp;
		occ_grid_->header.frame_id = "world";
		occ_grid_->info.map_load_time = cloud->header.stamp;
		grid_occupancy_pub_.publish(occ_grid_);

		/*
	// Go through cartesian grid(tmp_grid_)
	float y = -params_.grid_range_max + params_.grid_cell_size / 2;
	for(int j = 0; j < params_.width_grid; ++j, y += params_.grid_cell_size){

		float x = params_.grid_cell_size / 2;
		for(int i = 0; i < params_.height_grid; ++i, x += params_.grid_cell_size){
			
			//Buffer variables
			int seg, bin;

			//Get polar grid indices
			fromVeloCoordsToPolarCell(x, y, seg, bin);

			PolarCell & cell = polar_grid_[seg][bin];

			int cell_index = j * params_.height_grid + i;

			if(cell.p_logit == 0){
				tmp_grid_->data[cell_index] = -1;
			}
			else if(cell.p_logit < 0){
				tmp_grid_->data[cell_index] = 0;
			}
			else{
				tmp_grid_->data[cell_index] = 100;
			}
		}
	}

	// tmp_grid_->header.stamp = cloud->header.stamp;
	// tmp_grid_->header.frame_id = cloud->header.frame_id;
	// tmp_grid_->info.map_load_time = tmp_grid_->header.stamp;
	// tmp_occupancy_pub_.publish(tmp_grid_);
*/
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
		fromLocalGridToFinalCartesian(local_grid_x, local_grid_y, final_x, final_y);

		Vector4f new_vec = transMat * Vector4f(final_x, final_y, 0, 1);
		final_x = new_vec[0];
		final_y = new_vec[1];

		fromFinalCartesianToGridIndex(final_x, final_y, final_grid_index);
	}

	void SensorFusion::fromLocalGridToFinalCartesian(const int grid_x, const int grid_y,
													 float &x, float &y)
	{

		x = (grid_x + 0.5) * params_.grid_cell_size;
		y = (grid_y + 0.5 - params_.width_grid / 2) * params_.grid_cell_size;
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

	void SensorFusion::fixedPoint(const int point_x, const int point_y)
	{
		geometry_msgs::PointStamped p;
		p.header.frame_id = "world";
		p.header.stamp = cloud_stamp_;
		p.point.x = point_x;
		p.point.y = point_y;
		p.point.z = 0;
		fix_point_pub_.publish(p);
	}

	void SensorFusion::fromVeloCoordsToPolarInfo(const float x, const float y,
												 int &seg, int &bin, float &mag)
	{

		float ang = -std::atan2(y, x);
		mag = std::sqrt(x * x + y * y);
		seg = int((ang + params_.lidar_opening_angle) * params_.inv_angular_res);
		bin = int(mag * params_.inv_radial_res);

		// For last segment
		if (x == 0 && y < 0)
			seg = params_.grid_segments - 1;
	}

	void SensorFusion::fromVeloCoordsToPolarCell(const float x, const float y,
												 int &seg, int &bin)
	{

		float mag = std::sqrt(x * x + y * y);
		float ang = -std::atan2(y, x);
		seg = int((ang + params_.lidar_opening_angle) * params_.inv_angular_res);
		bin = int(mag * params_.inv_radial_res);

		// For last segment
		if (x == 0 && y < 0)
			seg = params_.grid_segments - 1;
	}

	void SensorFusion::fromPolarCellToVeloCoords(const int seg, const int bin,
												 float &x, float &y)
	{

		float mag = bin / params_.inv_radial_res + params_.grid_cell_size / 2;
		float ang = seg / params_.inv_angular_res - params_.lidar_opening_angle;
		y = -std::sin(ang) * mag;
		x = std::cos(ang) * mag;
	}
} // namespace sensor_processing
