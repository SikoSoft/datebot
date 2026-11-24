# CLAUDE.md

## Project

Datebot: learns the user's dating preferences from their own swipes and scores profiles 0–100, to eventually automate swiping on apps like Tinder, Hinge and Badoo. The model must be trainable over time and resettable at will.

## Git workflow (always follow)

- Work directly on `master` unless the user says otherwise.
- Commit every change in clean, separate, logically scoped commits (one concern per commit; never lump unrelated changes together).
- Always push to `origin master` after committing. Never leave commits unpushed.
- Write clear, imperative commit messages.

## Working style

- Ask the user clarifying questions on open decisions rather than guessing.
- Keep all profile data (other people's photos) local by default.
