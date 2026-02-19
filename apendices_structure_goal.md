```text
study_library/
|
|
|-- oop/
|   |-- 01_object_oriented_principles/
|   |   |-- 01_single_responsibility_principle.ipynb
|   |   |-- 02_open_closed_principle.ipynb
|   |   |-- 03_liskov_substitution_principle.ipynb
|   |   |-- 04_interface_segregation_principle.ipynb
|   |   |-- 05_dependency_inversion_principle.ipynb
|   |   |-- 06_cohesion.ipynb
|   |   `-- 07_coupling.ipynb
|   |
|   |-- 02_object_oriented_design_patterns/
|   |   |-- 01_factory_method_pattern.ipynb
|   |   |-- 02_strategy_pattern.ipynb
|   |   |-- 03_observer_pattern.ipynb
|   |   |-- 04_visitor_pattern.ipynb
|   |   |-- 05_adapter_pattern.ipynb
|   |   `-- 06_dependency_injection_pattern.ipynb
|   |
|   |-- 03_python_fundamentals/
|   |   |-- 01_python_execution_model.ipynb
|   |   |-- 02_scope_rules.ipynb
|   |   `-- 03_context_managers.ipynb
|   |
|   |-- 04_python_object_oriented_programming/
|   |   |-- 01_class_construction.ipynb
|   |   |-- 02_attribute_resolution.ipynb
|   |   |-- 03_inheritance.ipynb
|   |   |-- 04_composition.ipynb
|   |   `-- 05_protocols.ipynb
|   |
|   |-- 05_cpp_object_oriented_programming/
|   |   |-- 01_resource_acquisition_is_initialization.ipynb
|   |   |-- 02_unique_pointer.ipynb
|   |   |-- 03_shared_pointer.ipynb
|   |   |-- 04_rule_of_five.ipynb
|   |   |-- 05_virtual_dispatch.ipynb
|   |   `-- 06_type_erasure.ipynb
|   |
|   |-- 06_cpp_memory_for_objects/
|   |   |-- 01_object_layout.ipynb
|   |   |-- 02_alignment.ipynb
|   |   |-- 03_allocators.ipynb
|   |   `-- 04_polymorphic_memory_resource.ipynb
|   |
|   |-- 07_cpp_concurrency_for_object_oriented_systems/
|   |   |-- 01_memory_model.ipynb
|   |   |-- 02_atomics.ipynb
|   |   `-- 03_mutexes.ipynb
|   |
|   |-- 08_cpp_build_and_binary_compatibility/
|   |   |-- 01_compilation_pipeline.ipynb
|   |   |-- 02_linking_model.ipynb
|   |   |-- 03_symbol_visibility.ipynb
|   |   |-- 04_application_binary_interface.ipynb
|   |   `-- 05_one_definition_rule.ipynb
|   |
|   `-- 09_cpp_large_codebase_engineering/
|       |-- 01_layering_strategies.ipynb
|       |-- 02_interface_stability.ipynb
|       `-- 03_refactoring_strategy.ipynb
|
|-- llvm_compilers/
|   |
|   |-- 01_compiler_foundations/
|   |   |-- 01_control_flow_graph.ipynb
|   |   |-- 02_data_flow_analysis.ipynb
|   |   `-- 03_static_single_assignment_form.ipynb
|   |-- 02_llvm_intermediate_representation/
|   |   |-- 01_llvm_intermediate_representation_types.ipynb
|   |   |-- 02_llvm_intermediate_representation_instructions.ipynb
|   |   `-- 03_undefined_behavior_in_llvm.ipynb
|   |-- 03_pass_infrastructure/
|   |   |-- 01_new_pass_manager.ipynb
|   |   |-- 02_analysis_pass.ipynb
|   |   `-- 03_transformation_pass.ipynb
|   |-- 04_optimization_topics/
|   |   |-- 01_inlining.ipynb
|   |   |-- 02_dead_code_elimination.ipynb
|   |   `-- 03_loop_vectorization.ipynb
|   `-- 05_mlir/
|       |-- 01_mlir_dialects.ipynb
|       |-- 02_pattern_rewriting.ipynb
|       `-- 03_lowering_to_llvm.ipynb
|
|-- performance_and_systems/
|   |
|   |-- 01_central_processing_unit_performance/
|   |   |-- 01_cache_hierarchy.ipynb
|   |   |-- 02_translation_lookaside_buffer.ipynb
|   |   `-- 03_branch_prediction.ipynb
|   |-- 02_vectorization/
|   |   |-- 01_single_instruction_multiple_data_concepts.ipynb
|   |   `-- 02_vectorization_reports.ipynb
|   |-- 03_parallelism/
|   |   |-- 01_work_partitioning.ipynb
|   |   `-- 02_lock_contention.ipynb
|   `-- 04_profiling/
|       |-- 01_linux_perf_tool.ipynb
|       `-- 02_hardware_performance_counters.ipynb
|
|-- ml_infra/
|   |
|   |-- 01_ml_compiler_ecosystem/
|   |   |-- 01_openxla.ipynb
|   |   |-- 02_tensorflow_xla.ipynb
|   |   `-- 03_jax_compilation.ipynb
|   |-- 02_framework_integration/
|   |   |-- 01_tensorflow_graph_compilation.ipynb
|   |   |-- 02_jax_just_in_time_compilation.ipynb
|   |   `-- 03_custom_calls.ipynb
|   |-- 03_runtime_topics/
|   |   |-- 01_operator_fusion.ipynb
|   |   |-- 02_memory_planning.ipynb
|   |   `-- 03_dispatch.ipynb
|   `-- 04_benchmarking_and_regressions/
|       |-- 01_benchmark_design.ipynb
|       |-- 02_noise_control.ipynb
|       `-- 03_performance_regression_detection.ipynb
|
|-- open_source_workflow/
|   |
|   |-- 01_git_workflows/
|   |   |-- 01_rebasing.ipynb
|   |   |-- 02_bisecting.ipynb
|   |   `-- 03_submodules.ipynb
|   |-- 02_pull_requests/
|   |   |-- 01_pull_request_lifecycle.ipynb
|   |   `-- 02_review_comment_quality.ipynb
|   `-- 03_regression_triage/
|       |-- 01_minimal_reproducer.ipynb
|       `-- 02_test_reduction.ipynb
|
`-- interview_katas/
    |
    |-- 01_cpp_object_oriented_programming_katas/
    |   |-- 01_build_csv_reader_class_from_scratch.ipynb
    |   |-- 02_build_streaming_tokenizer_class_from_scratch.ipynb
    |   |-- 03_build_thread_safe_queue_class_from_scratch.ipynb
    |   |-- 04_build_object_pool_class_from_scratch.ipynb
    |   |-- 05_build_plugin_registry_class_from_scratch.ipynb
    |   |-- 06_build_dependency_injection_container_from_scratch.ipynb
    |   |-- 07_build_observer_pattern_from_scratch.ipynb
    |   |-- 08_build_strategy_pattern_from_scratch.ipynb
    |   |-- 09_build_type_erased_callback_wrapper_from_scratch.ipynb
    |   |-- 10_build_polymorphic_serializer_interface.ipynb
    |   |-- 11_build_raii_file_descriptor_wrapper.ipynb
    |   |-- 12_build_reference_counted_handle_class.ipynb
    |   |-- 13_build_pimpl_wrapper_class.ipynb
    |   |-- 14_fix_data_race_in_class_methods.ipynb
    |   `-- 15_reduce_lock_contention_in_class_design.ipynb
    |
    |-- 02_llvm_katas/
    |   |-- 01_write_function_pass_skeleton.ipynb
    |   |-- 02_register_pass_in_pipeline.ipynb
    |   |-- 03_write_analysis_pass_skeleton.ipynb
    |   |-- 04_emit_remarks_for_optimization.ipynb
    |   |-- 05_create_filecheck_test_from_output.ipynb
    |   |-- 06_write_lit_test_skeleton.ipynb
    |   |-- 07_minimize_miscompilation_reproducer.ipynb
    |   |-- 08_triage_optimization_regression_with_bisect.ipynb
    |   |-- 09_write_mlir_rewrite_pattern.ipynb
    |   `-- 10_write_mlir_conversion_pattern.ipynb
    |
    |-- 03_performance_katas/
    |   |-- 01_profile_hot_path_with_perf_tool.ipynb
    |   |-- 02_interpret_flamegraph.ipynb
    |   |-- 03_explain_cache_miss_regression.ipynb
    |   |-- 04_remove_dynamic_allocations_from_loop.ipynb
    |   |-- 05_fix_false_sharing_regression.ipynb
    |   |-- 06_vectorize_inner_loop.ipynb
    |   |-- 07_parallelize_reduction_kernel.ipynb
    |   |-- 08_build_microbenchmark_harness.ipynb
    |   `-- 09_establish_performance_regression_gate.ipynb
    |
    |-- 04_machine_learning_katas/
    |   |-- 01_build_tensor_class_from_scratch.ipynb
    |   |-- 02_build_shape_inference_from_scratch.ipynb
    |   |-- 03_build_broadcasting_rules_from_scratch.ipynb
    |   |-- 04_build_computational_graph_representation.ipynb
    |   |-- 05_build_topological_sort_scheduler.ipynb
    |   |-- 06_build_automatic_differentiation_engine.ipynb
    |   |-- 07_build_operator_fusion_pass.ipynb
    |   |-- 08_build_constant_folding_pass.ipynb
    |   |-- 09_build_common_subexpression_elimination_pass.ipynb
    |   |-- 10_build_layout_propagation_pass.ipynb
    |   |-- 11_build_kernel_dispatch_table.ipynb
    |   |-- 12_build_compilation_cache_key.ipynb
    |   `-- 13_build_minimal_machine_learning_library_architecture.ipynb
    |
    `-- 05_machine_learning_infrastructure_katas/
        |-- 01_build_reproducible_build_recipe_for_compiler.ipynb
        |-- 02_build_container_image_for_llvm_toolchain.ipynb
        |-- 03_build_ci_matrix_for_cpu_targets.ipynb
        |-- 04_build_ci_matrix_for_gpu_targets.ipynb
        |-- 05_build_build_cache_strategy_for_ci.ipynb
        |-- 06_build_artifact_promotion_pipeline.ipynb
        |-- 07_build_binary_compatibility_check_for_release.ipynb
        |-- 08_build_end_to_end_compilation_smoke_test.ipynb
        |-- 09_build_runtime_correctness_test_harness.ipynb
        |-- 10_build_determinism_check_for_compiled_artifacts.ipynb
        |-- 11_build_performance_benchmark_gate_for_pull_requests.ipynb
        |-- 12_build_minimal_reproducer_pipeline_for_regressions.ipynb
        |-- 13_build_symbol_visibility_audit_for_shared_libraries.ipynb
        |-- 14_build_dependency_pinning_strategy.ipynb
        `-- 15_build_telemetry_pipeline_for_compiler_and_runtime.ipynb
```
