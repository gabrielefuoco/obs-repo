
## Create a new repository

- Create a new directory.
- Open the directory.
- Run the following command to initialize a new Git repository:
   ```bash
   git init
   ```

## Checkout a repository

- To create a working copy of a local repository, run:
   ```bash
   git clone /path/to/repository
   ```
- When using a remote server, use the following command:
   ```bash
   git clone username@host:/path/to/repository
   ```

## Workflow

Your local repository consists of three "trees" maintained by Git:

- **Working Directory:** Holds the actual files.
- **Index:** Acts as a staging area.
- **HEAD:** Points to the last commit you've made.

![Git Trees](https://rogerdudler.github.io/git-guide/img/trees.png)

## Add & Commit

- **Propose changes (add to the Index):**
   ```bash
   git add <filename>
   ```
or
   ```bash
   git add *
   ```
- **Commit changes (to the HEAD):**
   ```bash
   git commit -m "Commit message"
   ```

## Pushing changes

- **Push changes to the remote repository:**
   ```bash
   git push origin master
   ```
Replace `master` with the desired branch name.

- **Connect to a remote server (if not cloned):**
   ```bash
   git remote add origin <server>
   ```
Now you can push changes to the selected remote server.

## Branching

Branches are used to develop features independently. The `master` branch is the default branch. Use other branches for development and merge them back to the `master` branch upon completion.

![Git Branches](https://rogerdudler.github.io/git-guide/img/branches.png)

- **Create a new branch and switch to it:**
   ```bash
   git checkout -b feature_x
   ```
- **Switch back to the `master` branch:**
   ```bash
   git checkout master
   ```
- **Delete the branch:**
   ```bash
   git branch -d feature_x
   ```
- **Push a branch to the remote repository:**
   ```bash
   git push origin <branch>
   ```

## Update & Merge

- **Update your local repository:**
   ```bash
   git pull
   ```
This fetches and merges remote changes.

- **Merge another branch into your active branch:**
   ```bash
   git merge <branch>
   ```
Git attempts to auto-merge changes. If conflicts arise, resolve them manually by editing the files indicated by Git. After editing, mark the files as merged:
   ```bash
   git add <filename>
   ```

- **Preview changes before merging:**
   ```bash
   git diff <source_branch> <target_branch>
   ```

## Tagging

Create tags for software releases.

- **Create a new tag:**
   ```bash
   git tag 1.0.0 1b2e1d63ff
   ```
Replace `1b2e1d63ff` with the first 10 characters of the commit ID you want to tag.

## Log

- **View repository history:**
   ```bash
   git log
   ```

- **Filter log output:**
- **Commits by a specific author:**
     ```bash
     git log --author=bob
     ```
- **Compressed log (one line per commit):**
     ```bash
     git log --pretty=oneline
     ```
- **ASCII art tree of branches:**
     ```bash
     git log --graph --oneline --decorate --all
     ```
- **Files changed in commits:**
     ```bash
     git log --name-status
     ```

For more options, see `git log --help`.

## Replace local changes

- **Replace local changes with the last HEAD content:**
   ```bash
   git checkout -- <filename>
   ```
This keeps changes added to the index and new files.

- **Drop all local changes and commits, fetch the latest history, and reset the local `master` branch:**
   ```bash
   git fetch origin
   git reset --hard origin/master
   ```

## Useful hints

- **Built-in Git GUI:**
   ```bash
   gitk
   ```
- **Colorful Git output:**
   ```bash
   git config color.ui true
   ```
- **One-line log format:**
   ```bash
   git config format.pretty oneline
   ```
- **Interactive adding:**
   ```bash
   git add -i
   ```

### Guides

- [Git Community Book](http://book.git-scm.com/)
- [Pro Git](http://progit.org/book/)
- [Think like a git](http://think-like-a-git.net/)
- [GitHub Help](http://help.github.com/)
- [A Visual Git Guide](http://marklodato.github.com/visual-git-guide/index-en.html)
