//! Python bindings for the Rust-based shader prediction system

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyReadonlyArray1};
use std::sync::Arc;
use std::path::PathBuf;

use ml_engine::{MLEngine, ShaderFeatures, FeatureExtractor, ShaderType};
use vulkan_cache::{VulkanShaderManager, CacheStats};

/// Python wrapper for the Rust ML predictor
#[pyclass]
struct RustMLPredictor {
    engine: Arc<MLEngine>,
    feature_extractor: FeatureExtractor,
}

/// Python wrapper for Vulkan shader cache
#[pyclass]
struct RustVulkanCache {
    manager: Arc<VulkanShaderManager>,
}

/// Python wrapper for shader features
#[pyclass]
#[derive(Clone)]
struct PyShaderFeatures {
    features: ShaderFeatures,
}

#[pymethods]
impl RustMLPredictor {
    #[new]
    fn new(model_path: Option<String>) -> PyResult<Self> {
        let engine = if let Some(path) = model_path {
            Arc::new(MLEngine::with_model(PathBuf::from(path))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        } else {
            Arc::new(MLEngine::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?)
        };
        
        Ok(Self {
            engine,
            feature_extractor: FeatureExtractor::new(),
        })
    }
    
    /// Predict compilation time for shader features
    fn predict_compilation_time(&self, py: Python, features_dict: &PyDict) -> PyResult<f32> {
        let features = self.extract_features_from_dict(features_dict)?;
        
        let result = py.allow_threads(|| {
            pyo3_asyncio::tokio::get_runtime()
                .block_on(self.engine.predict_compilation_time(&features))
        });
        
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Predict compilation times for multiple shaders
    fn predict_batch(&self, py: Python, features_list: &PyList) -> PyResult<Py<PyArray1<f32>>> {
        let mut features_vec = Vec::with_capacity(features_list.len());
        
        for item in features_list.iter() {
            let features_dict = item.downcast::<PyDict>()?;
            let features = self.extract_features_from_dict(features_dict)?;
            features_vec.push(features);
        }
        
        let predictions = py.allow_threads(|| {
            pyo3_asyncio::tokio::get_runtime()
                .block_on(self.engine.predict_batch(&features_vec))
        });
        
        let predictions = predictions
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyArray1::from_vec(py, predictions).to_owned())
    }
    
    /// Extract features from SPIR-V bytecode
    fn extract_features_from_spirv(&self, spirv_data: &[u8]) -> PyResult<PyShaderFeatures> {
        let features = self.feature_extractor.extract_from_spirv(spirv_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyShaderFeatures { features })
    }
    
    /// Extract features from shader source code
    fn extract_features_from_source(&self, source: &str, shader_type: &str) -> PyResult<PyShaderFeatures> {
        let shader_type = match shader_type.to_lowercase().as_str() {
            "vertex" => ShaderType::Vertex,
            "fragment" | "pixel" => ShaderType::Fragment,
            "geometry" => ShaderType::Geometry,
            "compute" => ShaderType::Compute,
            "tess_ctrl" | "tessellation_control" => ShaderType::TessellationControl,
            "tess_eval" | "tessellation_evaluation" => ShaderType::TessellationEvaluation,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown shader type: {}", shader_type)
            )),
        };
        
        let features = self.feature_extractor.extract_from_source(source, shader_type)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyShaderFeatures { features })
    }
    
    /// Add feedback for model improvement
    fn add_feedback(&self, features_dict: &PyDict, predicted: f32, actual: f32) -> PyResult<()> {
        let features = self.extract_features_from_dict(features_dict)?;
        self.engine.add_feedback(features, predicted, actual);
        Ok(())
    }
    
    /// Get prediction metrics and statistics
    fn get_metrics(&self) -> PyResult<PyDict> {
        let metrics = self.engine.get_metrics();
        let dict = PyDict::new(Python::acquire_gil().python());
        
        dict.set_item("cache_hit_rate", metrics.cache_hit_rate)?;
        dict.set_item("total_predictions", metrics.total_predictions)?;
        dict.set_item("cache_size", metrics.cache_size)?;
        dict.set_item("avg_inference_time_ns", metrics.avg_inference_time_ns)?;
        dict.set_item("fallback_usage_rate", metrics.fallback_usage_rate)?;
        
        Ok(dict.to_object(Python::acquire_gil().python()))
    }
    
    /// Extract shader features from Python dictionary
    fn extract_features_from_dict(&self, features_dict: &PyDict) -> PyResult<ShaderFeatures> {
        let mut features = ShaderFeatures::default();
        
        // Extract numeric features
        if let Some(val) = features_dict.get_item("instruction_count") {
            features.instruction_count = val.extract::<f32>()?;
        }
        if let Some(val) = features_dict.get_item("register_usage") {
            features.register_usage = val.extract::<f32>()?;
        }
        if let Some(val) = features_dict.get_item("texture_samples") {
            features.texture_samples = val.extract::<f32>()?;
        }
        if let Some(val) = features_dict.get_item("memory_operations") {
            features.memory_operations = val.extract::<f32>()?;
        }
        if let Some(val) = features_dict.get_item("control_flow_complexity") {
            features.control_flow_complexity = val.extract::<f32>()?;
        }
        if let Some(val) = features_dict.get_item("wave_size") {
            features.wave_size = val.extract::<f32>()?;
        }
        
        // Extract boolean features (converted to 0.0/1.0)
        if let Some(val) = features_dict.get_item("uses_derivatives") {
            features.uses_derivatives = if val.extract::<bool>()? { 1.0 } else { 0.0 };
        }
        if let Some(val) = features_dict.get_item("uses_tessellation") {
            features.uses_tessellation = if val.extract::<bool>()? { 1.0 } else { 0.0 };
        }
        if let Some(val) = features_dict.get_item("uses_geometry_shader") {
            features.uses_geometry_shader = if val.extract::<bool>()? { 1.0 } else { 0.0 };
        }
        
        // Extract Steam Deck specific features
        if let Some(val) = features_dict.get_item("thermal_state") {
            features.thermal_state = val.extract::<f32>()?;
        }
        if let Some(val) = features_dict.get_item("power_mode") {
            features.power_mode = val.extract::<f32>()?;
        }
        
        Ok(features)
    }
}

#[pymethods]
impl RustVulkanCache {
    #[new]
    fn new() -> PyResult<Self> {
        let manager = Arc::new(VulkanShaderManager::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?);
        
        Ok(Self { manager })
    }
    
    /// Enable Vulkan layer interception
    fn enable_interception(&mut self) -> PyResult<()> {
        // Note: This would need mutable access to the manager
        // In practice, we'd use interior mutability or redesign the API
        Ok(())
    }
    
    /// Get cache statistics
    fn get_stats(&self) -> PyResult<PyDict> {
        let stats = self.manager.get_stats();
        let dict = PyDict::new(Python::acquire_gil().python());
        
        dict.set_item("hit_rate", stats.hit_rate)?;
        dict.set_item("total_lookups", stats.total_lookups)?;
        dict.set_item("cache_hits", stats.cache_hits)?;
        dict.set_item("cache_misses", stats.cache_misses)?;
        dict.set_item("memory_usage_mb", stats.memory_usage_mb)?;
        dict.set_item("hot_tier_size", stats.hot_tier_size)?;
        dict.set_item("warm_tier_size", stats.warm_tier_size)?;
        dict.set_item("cold_tier_size", stats.cold_tier_size)?;
        
        Ok(dict.to_object(Python::acquire_gil().python()))
    }
}

#[pymethods]
impl PyShaderFeatures {
    #[new]
    fn new() -> Self {
        Self {
            features: ShaderFeatures::default(),
        }
    }
    
    /// Set instruction count
    fn set_instruction_count(&mut self, count: f32) {
        self.features.instruction_count = count;
    }
    
    /// Set register usage
    fn set_register_usage(&mut self, usage: f32) {
        self.features.register_usage = usage;
    }
    
    /// Set texture samples
    fn set_texture_samples(&mut self, samples: f32) {
        self.features.texture_samples = samples;
    }
    
    /// Set memory operations
    fn set_memory_operations(&mut self, ops: f32) {
        self.features.memory_operations = ops;
    }
    
    /// Set control flow complexity
    fn set_control_flow_complexity(&mut self, complexity: f32) {
        self.features.control_flow_complexity = complexity;
    }
    
    /// Set thermal state
    fn set_thermal_state(&mut self, state: f32) {
        self.features.set_thermal_state(state);
    }
    
    /// Set power mode
    fn set_power_mode(&mut self, mode: f32) {
        self.features.set_power_mode(mode);
    }
    
    /// Get complexity score
    fn complexity_score(&self) -> f32 {
        self.features.complexity_score()
    }
    
    /// Convert to dictionary
    fn to_dict(&self, py: Python) -> PyResult<PyDict> {
        let dict = PyDict::new(py);
        
        dict.set_item("instruction_count", self.features.instruction_count)?;
        dict.set_item("register_usage", self.features.register_usage)?;
        dict.set_item("texture_samples", self.features.texture_samples)?;
        dict.set_item("memory_operations", self.features.memory_operations)?;
        dict.set_item("control_flow_complexity", self.features.control_flow_complexity)?;
        dict.set_item("wave_size", self.features.wave_size)?;
        dict.set_item("uses_derivatives", self.features.uses_derivatives > 0.5)?;
        dict.set_item("uses_tessellation", self.features.uses_tessellation > 0.5)?;
        dict.set_item("uses_geometry_shader", self.features.uses_geometry_shader > 0.5)?;
        dict.set_item("shader_type_hash", self.features.shader_type_hash)?;
        dict.set_item("optimization_level", self.features.optimization_level)?;
        dict.set_item("cache_priority", self.features.cache_priority)?;
        dict.set_item("van_gogh_optimized", self.features.van_gogh_optimized > 0.5)?;
        dict.set_item("rdna2_features", self.features.rdna2_features > 0.5)?;
        dict.set_item("thermal_state", self.features.thermal_state)?;
        dict.set_item("power_mode", self.features.power_mode)?;
        
        Ok(dict)
    }
}

/// Utility functions
#[pyfunction]
fn is_steam_deck() -> bool {
    std::fs::read_to_string("/sys/devices/virtual/dmi/id/product_name")
        .map(|s| s.trim() == "Jupiter" || s.trim() == "Galileo")
        .unwrap_or(false)
    ||
    std::env::var("SteamDeck").is_ok()
    ||
    std::path::Path::new("/home/deck").exists()
}

/// Get system information for optimization
#[pyfunction]
fn get_system_info(py: Python) -> PyResult<PyDict> {
    let dict = PyDict::new(py);
    
    dict.set_item("is_steam_deck", is_steam_deck())?;
    dict.set_item("cpu_count", num_cpus::get())?;
    
    // Detect APU type on Steam Deck
    if is_steam_deck() {
        let apu_model = if std::path::Path::new("/sys/devices/virtual/dmi/id/board_name")
            .exists() 
        {
            std::fs::read_to_string("/sys/devices/virtual/dmi/id/board_name")
                .unwrap_or_default()
                .trim()
                .to_string()
        } else {
            "Unknown".to_string()
        };
        
        dict.set_item("apu_model", apu_model)?;
        dict.set_item("optimization_target", "steam_deck")?;
    } else {
        dict.set_item("optimization_target", "desktop")?;
    }
    
    Ok(dict)
}

/// Python module definition
#[pymodule]
fn shader_predict_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustMLPredictor>()?;
    m.add_class::<RustVulkanCache>()?;
    m.add_class::<PyShaderFeatures>()?;
    m.add_function(wrap_pyfunction!(is_steam_deck, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_info, m)?)?;
    
    // Add version information
    m.add("__version__", "3.0.0")?;
    
    Ok(())
}