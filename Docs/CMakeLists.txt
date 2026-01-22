install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DESTINATION .
      PATTERN "CMakeFiles*" EXCLUDE
      PATTERN "Makefile" EXCLUDE
      PATTERN "*.cmake" EXCLUDE
      )

add_custom_target(doc
      COMMAND ${CMAKE_COMMAND} -E env
      "${AIMET_PYTHONPATH}"
      "SW_VERSION=${SW_VERSION}"
      sphinx-build -v -T -b html ${CMAKE_BINARY_DIR}/Docs_SOURCE ${CMAKE_CURRENT_BINARY_DIR})

add_dependencies(doc
    copy_doc_source copy_examples)

add_custom_target(copy_doc_source
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/Docs_SOURCE)
