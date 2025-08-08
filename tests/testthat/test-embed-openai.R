test_that("azure openai embeddings works", {
  # testthat::skip_if(Sys.getenv("OPENAI_API_KEY") == "")
  testthat::skip_if(Sys.getenv("AZURE_OPENAI_API_KEY") == "")
  testthat::skip_if(Sys.getenv("AZURE_OPENAI_ENDPOINT") == "")

  ## Use the same model name for both APIs
  model <- "text-embedding-3-large"

  # --- OpenAI ---
  embs1_openai <- embed_openai("hello world", model = model)
  embs2_openai <- embed_openai("another hello world", model = model)

  embs_openai <- embed_openai(
    c("hello world", "another hello world"),
    model = model
  )

  expect_equal(embs1_openai[1, ], embs_openai[1, ])
  expect_equal(embs2_openai[1, ], embs_openai[2, ])

  # --- Azure OpenAI ---
  embs1_azure <- embed_azure_openai("hello world",
                                    deployment = model)
  embs2_azure <- embed_azure_openai("another hello world")

  embs_azure <- embed_azure_openai(
    c("hello world", "another hello world")
  )

  expect_equal(embs1_azure[1, ], embs_azure[1, ])
  expect_equal(embs2_azure[1, ], embs_azure[2, ])

  # Make sure shapes are identical between OpenAI and Azure OpenAI
  expect_equal(dim(embs_openai), dim(embs_azure))

  # Error when max tokens is reached
  expect_error(
    embed_openai(
      paste(rep(letters, 10000), collapse = ""),
      model = model
    ),
    regexp = "exceed the"
  )

  expect_error(
    embed_azure_openai(
      paste(rep(letters, 10000), collapse = "")
    ),
    regexp = "exceed the"
  )
})
