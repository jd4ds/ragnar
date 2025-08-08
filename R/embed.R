#' Embed Text
#'
#' @param x x can be:
#'  - A character vector, in which case a matrix of embeddings is returned.
#'  - A data frame with a column named `text`, in which case the dataframe is
#'    returned with an additional column named `embedding`.
#'  - Missing or `NULL`, in which case a function is returned that can be called
#'    to get embeddings. This is a convenient way to partial in additional arguments like `model`,
#'    and is the most convenient way to produce a function that can be passed to the `embed` argument of `ragnar_store_create()`.
#' @param base_url string, url where the service is available.
#' @param model string; model name
#' @param batch_size split `x` into batches when embedding. Integer, limit of
#'   strings to include in a single request.
#'
#' @returns If `x` is a character vector, then a numeric matrix is returned,
#'   where `nrow = length(x)` and `ncol = <model-embedding-size>`. If `x` is a
#'   data.frame, then a new `embedding` matrix "column" is added, containing the
#'   matrix described in the previous sentence.
#' @name embed_ollama
#' @examples
#' text <- c("a chunk of text", "another chunk of text", "one more chunk of text")
#' \dontrun{
#' text |>
#'   embed_ollama() |>
#'   str()
#'
#' text |>
#'   embed_openai() |>
#'   str()
#' }
NULL

#' @export
#' @rdname embed_ollama
embed_ollama <- function(
  x,
  base_url = "http://localhost:11434",
  model = "snowflake-arctic-embed2:568m",
  batch_size = 10L
) {
  if (missing(x) || is.null(x)) {
    args <- capture_args()
    fn <- partial(quote(ragnar::embed_ollama), alist(x = ), args)
    return(fn)
  }

  if (is.data.frame(x)) {
    x[["embedding"]] <- Recall(
      x[["text"]],
      base_url = base_url,
      model = model,
      batch_size = batch_size
    )
    return(x)
  }

  check_character(x)
  if (!length(x)) {
    # ideally we'd return a 0-row matrix, but currently the correct
    # embedding_size is not convenient to access in this context
    return(NULL)
  }

  starts <- seq.int(from = 1L, to = length(x), by = batch_size)
  ends <- c(starts[-1L] - 1L, length(x))

  embeddings <- map2(starts, ends, function(start, end) {
    req <- request(base_url) |>
      req_user_agent(ragnar_user_agent()) |>
      req_url_path_append("/api/embed") |>
      req_body_json(list(model = model, input = x[start:end])) |>
      req_error(body = \(resp) {
        resp_body_json(resp)$error
      })

    resp <- req_perform(req)
    resp_body_json(resp, simplifyVector = TRUE)$embeddings
  })

  list_unchop(embeddings)
}


#' @param api_key resolved using env var `OPENAI_API_KEY`
#' @param dims An integer, can be used to truncate the embedding to a specific size.
#' @param user User name passed via the API.
#'
#' @returns A matrix of embeddings with 1 row per input string, or a dataframe with an 'embedding' column.
#' @export
#' @rdname embed_ollama
embed_openai <- function(
    x,
    model = "text-embedding-3-small",
    base_url = "https://api.openai.com/v1",
    api_key = get_envvar("OPENAI_API_KEY"),
    dims = NULL,
    user = get_user(),
    batch_size = 20L
) {
  build_req <- function() {
    httr2::request(base_url) |>
      httr2::req_user_agent(ragnar_user_agent()) |>
      httr2::req_url_path_append("/embeddings") |>
      httr2::req_auth_bearer_token(api_key) |>
      httr2::req_retry(max_tries = 2L)
  }

  prepare_body <- function(texts, dims, user) {
    data <- list(model = model, input = as.list(texts))
    data$user <- user
    if (!is.null(dims)) {
      check_number_whole(dims, min = 1L)
      data$dimensions <- as.integer(dims)
    }
    data
  }

  .embed_openai_internal(
    x = x,
    dims = dims,
    user = user,
    batch_size = batch_size,
    build_req = build_req,
    prepare_body = prepare_body
  )
}

#' @param api_key resolved using env var `OPENAI_API_KEY`
#' @param dims An integer, can be used to truncate the embedding to a specific size.
#' @param user User name passed via the API.
#'
#' @returns A matrix of embeddings with 1 row per input string, or a dataframe with an 'embedding' column.
#' @export
#' @rdname embed_ollama
embed_azure_openai <- function(
    x,
    deployment,
    api_version = "2024-02-15-preview",
    base_url = get_envvar("AZURE_OPENAI_ENDPOINT"),
    api_key = get_envvar("AZURE_OPENAI_API_KEY"),
    dims = NULL,
    user = get_user(),
    batch_size = 20L
) {
  build_req <- function() {
    httr2::request(base_url) |>
      httr2::req_user_agent(ragnar_user_agent()) |>
      httr2::req_url_path_append(sprintf("/openai/deployments/%s/embeddings", deployment)) |>
      httr2::req_url_query(`api-version` = api_version) |>
      httr2::req_headers(`api-key` = api_key) |>
      httr2::req_retry(max_tries = 2L)
  }

  prepare_body <- function(texts, dims, user) {
    data <- list(input = as.list(texts))
    data$user <- user
    if (!is.null(dims)) {
      check_number_whole(dims, min = 1L)
      data$dimensions <- as.integer(dims)
    }
    data
  }

  .embed_openai_internal(
    x = x,
    dims = dims,
    user = user,
    batch_size = batch_size,
    build_req = build_req,
    prepare_body = prepare_body
  )
}


.embed_openai_internal <- function(
    x,
    dims = NULL,
    user = get_user(),
    batch_size = 20L,
    max_tokens = 8191L,
    build_req,
    prepare_body
  ) {
    if (missing(x) || is.null(x)) {
      args <- capture_args()
      fn <- partial(.embed_openai_internal, alist(x = ), args)
      return(fn)
    }

    if (is.data.frame(x)) {
      x[["embedding"]] <- Recall(
        x[["text"]],
        dims = dims,
        user = user,
        batch_size = batch_size,
        max_tokens = max_tokens,
        build_req = build_req,
        prepare_body = prepare_body
      )
      return(x)
    }

    text <- x
    check_character(text)

    if (!length(text)) {
      # ideally we'd return a 0-row matrix, but currently the correct
      # embedding_size is not convenient to access in this context
      return(NULL)
    }

    ## open ai models have max token length of 8191... what happens if too long?
    token_lengths <- tryCatch(ragnar::num_tokens(text), error = function(e) NA_integer_)
    too_long <- which(!is.na(token_lengths) & token_lengths > max_tokens)
    if (length(too_long)) {
      stop(sprintf(
        "The following inputs exceed the %d token limit: %s",
        max_tokens,
        paste(too_long, collapse = ", ")
      ), call. = FALSE)
    }

    starts <- seq.int(from = 1L, to = length(text), by = batch_size)
    ends <- c(starts[-1L] - 1L, length(text))

    embeddings <- map2(starts, ends, function(start, end) {
      body <- prepare_body(text[start:end], dims, user)

      req <- build_req() |>
        httr2::req_body_json(body)

      resp <- httr2::req_perform(req)

      # embeddings is a list of length(text), of double vectors
      resp_body_json(resp, simplifyVector = TRUE)$data$embedding
    })

    matrix(unlist(embeddings), nrow = length(text), byrow = TRUE)
  }


# ---- utils ----

get_envvar <- function(name, error_call = caller_env()) {
  val <- Sys.getenv(name, NA_character_)
  if (is.na(val)) {
    if (is_testing()) {
      testthat::skip(sprintf("%s env var is not configured", name))
    } else {
      cli::cli_abort("Can't find env var {.code {name}}.", call = error_call)
    }
  }
  val
}

get_user <- function() {
  sys_info <- Sys.info()
  user <- sys_info[["effective_user"]]
  if (user != "unknown") {
    return(user)
  }
  user <- sys_info[["user"]]
  if (user != "unknown") {
    return(user)
  }
  NULL
}

ragnar_user_agent <- function() {
  paste0("r-ragnar/", .package_version)
}

is_testing <- function() {
  identical(Sys.getenv("TESTTHAT"), "true")
}

.package_version <- c(read.dcf('DESCRIPTION', 'Version'))
