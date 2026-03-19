test_that("grecov_tail returns valid tail probabilities", {
  result <- grecov_tail(
    p = c(0.3, 0.7),
    v = c(0, 1),
    s_obs = 8,
    n = 10L
  )

  expect_type(result, "list")
  expect_true(result$prob_left > 0 && result$prob_left <= 1)
  expect_true(result$prob_right > 0 && result$prob_right <= 1)
  expect_true(result$explored_mass > 0.99)
  expect_true(result$states_explored > 0)
})

test_that("grecov_tail with uniform distribution", {
  result <- grecov_tail(
    p = c(0.5, 0.5),
    v = c(0, 1),
    s_obs = 5,
    n = 10L
  )

  # For symmetric case, both tails should be roughly equal
  expect_true(abs(result$prob_left - result$prob_right) < 0.1)
})
