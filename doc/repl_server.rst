.. _repl_server:

Serving the REPL
================

The REPL page (:file:`doc/_static/repl`) runs in a browser.  NeuroLang
answers it in one of two places:

- **in the browser**, on Pyodide, from a bundle of NeuroLang's wheels and
  an engine's data: the page of the documentation, or
  ``neurolang-repl-serve`` without ``--local``;
- **on a server**, in its Python, with ``neurolang-repl-serve --local``:
  the engines with all their analyses, as fast as the machine, and, with a
  users file, a session of its own for each tab of each user, the AI
  assistant's keys kept on the server, an admin console and the ratings
  stored with the user who rates.

This page is about the second: how to set up such a server for a group of
people.  :doc:`contributing` tells how to run the page of a checkout for
development.

.. contents:: On this page
   :local:
   :depth: 1


Quick start
-----------

On a Linux server, from a checkout of NeuroLang, with `uv
<https://docs.astral.sh/uv/>`_::

   git clone https://github.com/NeuroLang/NeuroLang && cd NeuroLang
   uv sync --extra apps --extra tui --extra llm

   # the first user, an admin (asks their password; makes users.yaml)
   uv run --extra apps neurolang-repl-serve --users users.yaml --add-user ann --groups admin

   # its options in server.yaml, its secrets in secrets.yaml (chmod 600):
   # see "Configuration files" below
   cat > server.yaml <<EOF
   local: true
   host: 0.0.0.0
   port: 8443
   cert: cert.pem
   key: key.pem
   users: users.yaml
   feedback_db: feedback.sqlite
   activity_db: activity.sqlite
   data_dir: /srv/neurolang_data
   secrets: secrets.yaml
   EOF

   # serve it, as a service of this user, over HTTPS
   uv run --extra apps --extra tui --extra llm neurolang-repl-serve \
       --daemon install --config server.yaml

   loginctl enable-linger $USER   # keep it running after you log out

Then open ``https://SERVER:8443/``, log in as ``ann``, and open the admin
console at ``https://SERVER:8443/admin`` to add the other users.  Before
serving, the server prepares each engine: it downloads its data into
``--data-dir`` and finds the tables it takes long to find, kept there, which
takes a few minutes the first time and is quicker after; an engine it
cannot prepare (a source down) is told in the log, and its first session
tries again.  A session then starts from the data kept, without the
network: the first one of an engine takes the time of a build (about a
minute for Neurosynth with ``--light``), the next ones under a second.


Requirements
------------

- Python 3.12 or later, and the workspace's extras ``apps``, ``tui`` (the
  REPL) and ``llm`` (the AI assistant): ``uv sync --extra apps --extra tui
  --extra llm``.  Use the same extras with every ``uv sync`` in that
  checkout: one without them removes their packages from its environment,
  which the service runs, and the service then fails to start (``uv run``
  without them leaves the packages there: it installs what it lacks, and
  removes nothing).
- Memory for the engines: several GB for Neurosynth with all its analyses
  (``lift``, ``relative_risk``...; ``--light`` leaves them out, as the page
  in the browser does).  The sessions of the users share the memory of the
  engine's data (see `Sessions`_).
- On Linux, forking the sessions (``--users``); macOS works too.
- Access to the internet from the server, for the first download of the
  data and for the AI providers.


The command
-----------

``neurolang-repl-serve`` is the command of the workspace's root project
(:file:`tools/serve_repl`); ``python -m serve_repl`` is the same.  Its
options, by what they do:

Serving
~~~~~~~

``--local``
   NeuroLang runs in this server's Python; the page's worker sends what it
   is asked to the server (``/api/...``).  Without it, the server serves the
   page and a bundle for the browser (see :doc:`contributing`).
``--data-dir DIR``
   The engines' data directory (default ``neurolang_data``, relative to
   where the command runs; give an absolute path for a service).
``--no-prepare``
   Do not prepare the engines before serving; the first session of each
   downloads what it lacks.
``--no-preload-embeddings``
   Do not load an engine's text embeddings when it is built (with
   ``--users``, before forking its sessions): its first query of them
   does; see `Sessions`_.
``--resolution MM``
   The resolution of the engines' MNI grid (default 2).
``--light``
   Leave out the analyses a browser cannot hold, as the page does.
``--engine NAME``
   The engines the page offers (repeat it; default: all of them).
``--host HOST``, ``--port PORT``
   Where it listens: ``127.0.0.1`` by default, this machine only; ``0.0.0.0``
   for the network.  A port below 1024 needs root, which the service does
   not have: use a port above, or a reverse proxy.
``--server-name NAME``
   Without ``--users`` nor ``--password-file``, the server answers only the
   requests sent to (whose ``Host`` is) ``localhost``, ``127.0.0.1``,
   ``::1``, ``--host`` (unless ``0.0.0.0`` or ``::``) and the names of
   ``--server-name`` (repeat it): another site whose name it made lead here
   (DNS rebinding) would otherwise use it.  Give the name, or the address,
   by which the other machines reach it.  With users or a password, any
   name is answered.
``--site DIR``
   Where the page is put (default :file:`~/.cache/neurolang/repl-site`).
``--no-browser``
   Do not open the page in a browser (the service never does).

HTTPS
~~~~~

``--cert CERT.pem``, ``--key KEY.pem``
   Serve HTTPS with this certificate (its chain after it) and its private
   key, which may be in the certificate's file.  Without a certificate, a
   password is refused unless the server listens on this machine only, as
   it would go in clear.
``--key-password-file FILE``
   The passphrase of the key, if it has one, on the first line of FILE
   (readable by you only); without it, the passphrase is asked in the
   terminal, which a service has not.

If they cannot serve, the server says why: the certificate's file has no
key (give ``--key``), the key is not that of the certificate, the passphrase
is not that one, a file is missing, or they are not PEM.  Without users nor
a password, give the name of the certificate with ``--server-name`` too
(see `Serving`_).  To check them::

   head -1 cert.pem    # -----BEGIN CERTIFICATE-----
   head -1 key.pem     # -----BEGIN PRIVATE KEY----- (or RSA / EC PRIVATE KEY);
                       # -----BEGIN ENCRYPTED PRIVATE KEY-----: it has a passphrase
   openssl x509 -in cert.pem -noout -subject -enddate
   openssl pkey -in key.pem -noout && echo "the key reads"
   # the key is the certificate's if these two are the same:
   openssl x509 -in cert.pem -noout -pubkey | openssl sha256
   openssl pkey -in key.pem -pubout | openssl sha256

A binary (DER) file converts to PEM with ``openssl x509 -inform DER -in
cert.der -out cert.pem`` (``openssl pkey -inform DER`` for the key); a
``.p12`` or ``.pfx`` bundle with ``openssl pkcs12 -in bundle.p12 -clcerts
-nokeys -out cert.pem`` and ``openssl pkcs12 -in bundle.p12 -nocerts -out
key.pem``.

A certificate of `Let's Encrypt <https://letsencrypt.org/>`_ (``certbot
certonly --standalone -d SERVER``) works:
``--cert /etc/letsencrypt/live/SERVER/fullchain.pem --key
/etc/letsencrypt/live/SERVER/privkey.pem``, readable by the user of the
service.  The server reads them when it starts: restart it after a renewal
(``--daemon reinstall``, or ``systemctl --user restart neurolang-repl``).

Who may use it
~~~~~~~~~~~~~~

``--users users.yaml``
   The users, each with a name and a password, their groups, and their
   assistants (see `Users and groups`_); with ``--local``, each tab of each
   user has a session of its own.
``--password-file FILE``
   Instead of users, one password (the first line of FILE), with any user
   name; the visitors share one session.
``--max-sessions N``, ``--idle-minutes M``
   With ``--users`` and ``--local``: the sessions at most at once (8, at
   least 1), and the minutes after which an unused one ends (120, above 0).
``--max-running N``, ``--query-minutes M``
   With ``--users`` and ``--local``: the queries running at once, of all
   the sessions (the machine's CPUs, at least 1), and the minutes after
   which a query is stopped (60; 0, never); see `Sessions`_.
``--guests``, ``--max-guest-sessions N``
   With ``--users`` and ``--local``: let visitors in as guests, without an
   account, at most N sessions of guests at once (2); see `Guests`_.
``--secure-cookie``
   With ``--users`` behind a reverse proxy of HTTPS (no ``--cert``): the
   login cookie goes over HTTPS only, as it does with ``--cert``.
``--orcid orcid.yaml``
   With ``--users``: the users named by an ORCID iD sign in with ORCID, and
   an iD of no user may ask for an account (see `Signing in with ORCID`_).

Configuration
~~~~~~~~~~~~~

``--config server.yaml``, ``--secrets secrets.yaml``
   The options, and the secrets, in YAML files (see `Configuration files`_).

Without either, anyone who reaches the server uses it.

The AI assistant
~~~~~~~~~~~~~~~~

``--env-file llm.env``
   ``NAME=VALUE`` lines put in the server's environment: the key of the
   provider of the server's AI configuration (see `The AI assistant's
   keys`_), the default of those who log in.  It needs ``--local``, and
   ``--users`` or ``--password-file``.

Storing the ratings
~~~~~~~~~~~~~~~~~~~

``--feedback-db feedback.sqlite``
   With ``--local``, store the ratings of the page on the server (see
   `Feedback`_).

``--activity-db activity.sqlite``
   With ``--local`` and ``--users``, log what each user runs, its result
   and the resources it took (see `Activity and usage`_), which the admins
   read at ``/admin/activity`` (see `The activity page`_).

``--dashboard``
   With ``--activity-db``, Datasette on the log and the ratings, for the
   admins, at ``/admin/logs/``: every table and any read-only SQL (see
   `Raw data and SQL`_).

Running as a service
~~~~~~~~~~~~~~~~~~~~

``--daemon install|reinstall|remove|status``, ``--python PATH``
   Install the server as a service of this user, or tell how it runs (see
   `The service`_ and `Uptime`_).

Managing the users
~~~~~~~~~~~~~~~~~~

``--list-users``, ``--add-user``, ``--set-password``, ``--set-groups``,
``--set-llm``, ``--set-group-llm``, ``--own-key``, ``--set-key``,
``--set-group-key``, ``--remove-user`` and ``--hash-password`` change the
users file from the command line (see `Managing users`_).


Configuration files
-------------------

The server's options go in a YAML file, ``--config server.yaml``, each
by its name (``data_dir``, or ``data-dir`` as the option is written), a list
for an option given several times (``engines: [neurosynth, neurostore]``),
``true`` for a flag; a relative path is relative to the file.  An option
given on the command line replaces the file's.  The service installed with
``--config`` runs that file as it is, so that a change of it is the
service's at its next start (``systemctl --user restart neurolang-repl``).
The users, their groups and their assistants stay in their own file
(``users``), which the console writes.  The options of the sessions, their
log and the dashboard go there too::

   max_sessions: 8
   idle_minutes: 120
   max_running: 4              # queries at once (the machine's CPUs)
   query_minutes: 60           # 0: no limit
   activity_db: activity.sqlite
   dashboard: true             # needs the dashboard extra

Its secrets go in a second YAML file, readable by you only (the server
refuses it otherwise: ``chmod 600 secrets.yaml``), which ``server.yaml``
names (``secrets: secrets.yaml``), or ``--secrets FILE``::

   orcid:                      # the client of ORCID (see Signing in with ORCID)
     client_id: APP-XXXXXXXXXXXXXXXX
     client_secret: ...
   env:                        # the server's variables (--env-file)
     ALBERT_API_KEY: ...
   password: ...               # the one password of all (--password-file)
   key_password: ...           # the passphrase of the key of cert (--key-password-file)

so that ``server.yaml`` holds no secret and may be shared or versioned.
Each secret may still be given in a file of its own, as before
(``--orcid``, ``--env-file``, ``--password-file``,
``--key-password-file``), but not both ways at once.


Users and groups
----------------

The users file is YAML, readable by its owner only (``chmod 600``; the
server and the console keep it so)::

   users:
     ann:
       password: scrypt$16384$8$1$...    # made by --add-user, --set-password
       groups: [admin, lab]
     bob:
       password: scrypt$16384$8$1$...
       groups: [lab]
       llm:                              # bob's own assistant
         provider: anthropic
         model: claude-sonnet-5-5
         key: sk-ant-...
     eve:
       password: scrypt$16384$8$1$...
       groups: [guest]
       own_key: true                     # eve may give her own key
   groups:
     lab:
       llm:                              # the assistant of the lab's users
         provider: openrouter
         model: anthropic/claude-sonnet-5.5
         key: sk-or-...
       own_key: false                    # the lab's users may not give theirs
     guest: {}

A user has:

``password``
   The scrypt hash of their password (never the password).
``groups``
   Labels.  Those of ``admin`` have the admin console; the others are for
   the assistants (below) and are recorded with each rating.
``llm``
   Their own assistant: a ``provider`` of the page's form (``anthropic``,
   ``openai``, ``openrouter``, or ``custom``, an OpenAI-compatible API at
   ``base_url``, which it needs), its ``model`` (the provider's by default)
   and its ``key``.  A change that gives no key keeps the one before, for
   the same provider and ``base_url`` only.
``own_key``
   Whether they may give their own key, in the page and on their account
   page.
``env``
   Variables of their sessions: a key for the server's AI configuration
   (see `The AI assistant's keys`_).

A group has ``llm``, ``own_key`` and ``env`` too, which its users have.

The server reads the file again when it changes: an edit by hand, or by the
command line, reaches it without a restart, and the open pages of a user
take a new assistant at their next start (the console's and the account
page's changes reach them at once).  The console and the command line write
the file anew (its comments are lost), holding a lock of it
(:file:`users.yaml.lock`), so that their changes all stay; they refuse to
write over a file edited by hand that the server cannot read (it says
why), and keep a user in it.  A file of the users alone, without ``users:`` and
``groups:``, as an older server wrote it, is read as well.


The AI assistant's keys
-----------------------

The assistant of a session answers with, in order:

1. the key the user gives in the page (the *API key* button), if they may
   give their own;
2. their own ``llm``;
3. the ``llm`` of their groups: the first, by name, that has one;
4. the server's AI configuration, :file:`~/.config/neurolang/llm.yaml`
   (see :doc:`query_assistant`), with the key in the variable its provider
   names (``api_key_env``, e.g. ``ALBERT_API_KEY``), taken from the user's
   ``env``, else their groups', else the server's ``--env-file``;
5. none: the page asks the user for a key.

A user may give their own key (1, and saving their ``llm`` on their account
page) as their ``own_key`` says, else not if one of their groups says
``own_key: false``, else they may.  When they may not, the page hides its
*API key* button and the server refuses a key from it.

The keys stay on the server: the page, the console and the account page
never show one; a session has its user's key in the environment of its own
process only.  An admin who wants to know which key a user has looks at the
console's *AI assistant* column: theirs, their group's (which group), the
server's, or none.


The admin console
-----------------

``/admin`` is for the users of the group ``admin``; anyone else gets a
refusal (403).  It shows:

- **Users**: their groups, their assistant (whose key: their own, their
  group's, the server's or none; its provider and model), their variables'
  names, whether they may give their own key.  It adds a user, changes their
  password, groups or own key, sets or removes their assistant, and removes
  them (ending their sessions).  The last admin cannot be removed.
- **Groups**: their members, assistant, variables and own key; it sets or
  removes a group's assistant and own key.
- **Sessions**: the user, tab and engine of each, how long it has been
  unused, whether it works, the CPU seconds and the memory of its process
  (see `Activity and usage`_); it ends one.  Above them, the memory of the
  builder, which the sessions share.
- **Usage by user**: with ``--activity-db``, each user's inputs run and
  failed, their time and CPU seconds, the most memory of their sessions,
  and a link to their log.
- **Feedback**: the latest 200 rated answers, with the user and their
  groups, and a link for each engine exporting the rated answers of its
  assistant as its examples, in YAML.

A change applies at once to the file and to the open pages of the users
it changes.  A password has 8 characters at least, here, on the account
page and from the command line.


Logging in
----------

With ``--users``, the server has a login page, ``/login``: a page opened
without a login leads there, and back once logged in (to a page of the
server only).  A login is a cookie signed by the server (HttpOnly,
SameSite=Lax, Secure over HTTPS or with ``--secure-cookie``), good for 7
days; it ends when the user logs out (*Log out*, at the top of the REPL
with ``--local``, ``/logout``), in every browser where they logged in
before (every copy of the cookie),
when their password changes (the browser that changed it stays logged in),
when they are removed, and when the server restarts (it signs with a key
made at its start).  Logging out, and opening the login page, forget the
API keys the REPL kept in the browser.  A page whose login expired opens
the login page.  Scripts may still give a user's name
and password with HTTP Basic (``curl -u ann:...``).  With
``--password-file``, the browser asks the password itself, and cannot log
out.

Signing in with ORCID
---------------------

With ``--orcid orcid.yaml``, the login page offers *Sign in with ORCID*.
The server sends the visitor to ORCID's page, where they sign in, and ORCID
sends them back with a code, which the server exchanges for their ORCID iD
and their name (OAuth 2.0 with the scope ``/authenticate``: who they are,
nothing else of their record).

- A user whose name is that iD logs in, as with a password.  Such a user
  needs no password (``password`` is left out of their entry), and their
  name at ORCID is kept as their ``orcid_name``, which the REPL, the console
  and their account page show::

     users:
       0000-0002-1825-0097:
         groups: [lab]
         orcid_name: Josiah Carberry

- An iD of no user may **ask for an account**, with a message for the
  admins.  The requests wait in :file:`users.yaml.requests.json`, next to the
  users file; the console lists them under *Account requests*, where an
  admin approves one, adding the user named by the iD to the groups given,
  or declines it.  Until then, the iD has no access.

The client is ORCID's: register one at `ORCID's developer tools
<https://orcid.org/developer-tools>`_ (its *Public API* is enough), with the
redirect URI ``https://SERVER:PORT/login/orcid/callback``, and give its id
and secret as the ``orcid`` of the secrets (see `Configuration files`_), or
in a YAML file of its own, ``--orcid orcid.yaml`` (readable by you only)::

   client_id: APP-XXXXXXXXXXXXXXXX
   client_secret: ...
   # when the server is reached at another address than its requests name
   # (behind a proxy), the redirect URI registered:
   redirect_uri: https://repl.example.org/login/orcid/callback
   # a client of ORCID's sandbox, for testing:
   sandbox: false

From the command line::

   neurolang-repl-serve --users users.yaml --add-user 0000-0002-1825-0097 --groups lab  # no password
   neurolang-repl-serve --users users.yaml --list-requests
   neurolang-repl-serve --users users.yaml --approve-request 0000-0002-1694-233X --groups lab
   neurolang-repl-serve --users users.yaml --decline-request 0000-0002-1694-233X

In the console, *Add* with an iD as the user's name and no password adds a
user who signs in with ORCID.

**Merging and separating.**  A user of a name of their own may sign in with
an ORCID iD too: merging the ORCID user named by the iD into them (the
console's *merge into…*, or ``--merge-orcid ID USER``) gives them
``orcid: ID``, the groups of both, and the ORCID user's assistant,
variables and own key where they have none; the ORCID user is no more, and
their sessions end.  They then log in with their password, or with ORCID::

   users:
     bob:
       password: scrypt$16384$8$1$...
       groups: [imaging, lab]
       orcid: 0000-0002-1694-233X
       orcid_name: Sofia Garcia

Users do it themselves on their account page too:

- a user with a password links their iD (*Link my ORCID iD*): they sign in
  at ORCID, which proves the iD theirs, and log in with either then; an
  iD with an account of its own is merged into theirs, as above.  They
  unlink it there (*Unlink it*), and log in with their password only.
- a user who signs in with ORCID only merges their account into one of
  theirs with a password (*Merge into it*, giving its name and password):
  they are logged in as it then.

An iD that signs in another user is not linked: an admin decides.

Separating it (*Separate ORCID*, or ``--separate-orcid USER``) makes the iD a
user of its own again, named by it, in the user's groups but ``admin``,
which an admin gives again if it is theirs.  An iD signs in one user only:
the server refuses a users file where it names a user and is another's
``orcid``.


Guests
------

With ``--guests``, the login page offers *Continue as a guest*: a visitor
comes in without an account, as a guest of a name of its own (``guest:``
and a random id, shown as *guest*), with sessions of their own, in the group
``guest``.  A guest's AI assistant answers only with a key they give in the
page: their sessions have no key of the server, of a group or of a user
(the server's keys are not even in their processes' environment).  A guest
has no account page, no console, and their ratings are signed by their
guest's name and the group ``guest``.  The guests have at most
``--max-guest-sessions`` sessions at once (2), so that they cannot take
those of the users.  A guest's login ends when they log out, after 7
days, and when the server restarts; a guest who logs in again is another
guest, with sessions of their own.

Accounts
--------

With ``--local``, the REPL shows the name of the user logged in at its top,
linked to their account page, ``/account``: their name and groups, whose
assistant they have, a form that saves their own (provider, model, its
address for ``custom``, key; the key kept on the server and never shown
again; it answers in their open pages at once), if they may give their own,
and a form that changes their password (the current one first; the new one
of 8 characters at least).  A forgotten password is reset by an admin, in
the console or with ``--set-password``.


Managing users
--------------

The same changes as the console, from the command line, on the file of
``--users``; each asks a password or a key without echoing it (prefix them
with ``uv run --extra apps`` in a checkout)::

   neurolang-repl-serve --users users.yaml --add-user ann --groups admin,lab   # makes the file
   neurolang-repl-serve --users users.yaml --set-password ann
   neurolang-repl-serve --users users.yaml --set-groups bob --groups lab
   neurolang-repl-serve --users users.yaml --set-llm bob anthropic claude-sonnet-5-5   # empty key: removes it
   neurolang-repl-serve --users users.yaml --set-group-llm lab openrouter anthropic/claude-sonnet-5.5
   neurolang-repl-serve --users users.yaml --own-key group:lab no       # or NAME yes|no|inherit
   neurolang-repl-serve --users users.yaml --set-key bob ALBERT_API_KEY # a variable of env
   neurolang-repl-serve --users users.yaml --set-group-key lab ALBERT_API_KEY
   neurolang-repl-serve --users users.yaml --remove-user bob
   neurolang-repl-serve --users users.yaml --list-users                 # no secrets

``--hash-password`` prints the hash of a password, for a file written by
hand.


Sessions
--------

With ``--users`` and ``--local``, each tab of each user has a session of its
own, in a process of its own (``neurolang_apps.utils.tui.sessions``):

- A *builder* process, started anew before the server answers, builds each
  engine once (the first start of an engine takes the time of a build; the
  next start in another tab takes under a second), then forks a process per
  session from it.  The sessions share the memory of the engine's data, as
  forked processes do until they write it.
- Before its first fork, the builder loads the engine's text embeddings
  (those of its documents, from their cache, and the query model), which
  its sessions then share, rather than each load them, in memory of its
  own, at its first ``query_similarity`` (unless
  ``--no-preload-embeddings``, or ``preload_embeddings: false``; the
  single-user ``--local`` server loads them when it builds an engine, and
  preparing the engines writes their cache).  The query model is the numpy one
  (``specter2_numpy``) when its weights are next to the model's copies,
  ``{data_dir}/{engine}/embeddings/models/*.npz`` (for an engine extending
``neurosynth`` or ``neurostore``, those of the engine it extends): it loads in half a
  second, without torch.  Otherwise it is torch's, on the CPU, which needs
  the ``embeddings`` extra, as computing the documents' embeddings does
  when their cache is missing (hours on a CPU: ``python -m
  neurolang.utils.text_embeddings --engine NAME`` does it ahead of time,
  best on a GPU).  What cannot be loaded is logged, and loaded by each
  session at its first query, as before.
- Each session has its own rules, assistant, queries (which run at the same
  time as those of the others) and files; *Stop* and *Restart* end that
  session's process only, and the page starts it again.
- A session is that of a user and a tab: another user cannot reach it.  A
  reload keeps the tab's session; a duplicated tab has one of its own.
- At most ``--max-sessions`` run at once: past them, the one unused the
  longest ends, unless it is at work or was used in the last minute (then
  the new tab is told to wait).  One unused for ``--idle-minutes`` ends.
- A session the server ended (unused, made room for another, its query ran
  too long) comes back at the next input of its page: the page starts it
  again, runs again the rules the session defined (as after *Stop*, which
  the page keeps; not its queries, nor the questions to the assistant),
  then that input.  If it ends again while its rules come back, or at once
  after, the page says so rather than try again.  A restart of the server
  ends the logins too: the page opens the login page, then starts anew,
  without the rules of the session.  With ``--activity-db``, the rules run
  again are logged again, as inputs.
- At most ``--max-running`` queries (and downloads of results) run at once,
  of all the sessions: one more waits its turn, its page saying so, so that
  the queries of many users share the machine's CPUs rather than slow each
  other down.  A session that is not running a query costs no CPU, only its
  memory.
- A query running longer than ``--query-minutes`` is stopped: its session's
  process ends (which frees its CPU and memory at once, as no thread of
  Python can be stopped otherwise), and its page says so; its rules come
  back at the next input.

Without ``--users``, one session serves everyone, in the server's process.


Activity and usage
------------------

The server takes, every 10 seconds, the CPU seconds and the memory of each
session's process and of the builder, which the console shows.  A
session's memory is its own, *private* memory: what it adds to the
machine's beyond the engine's data, which it shares with the builder (its
resident memory, ``rss``, counts those shared pages, and so overstates it).
The builder's resident memory is that of the engines' data.  This is read
from :file:`/proc` and so on Linux only; elsewhere the console shows ``-``.
Together they tell how many sessions a machine holds: about (its memory −
the builder's) / a session's own, at its most.

With ``--activity-db activity.sqlite``, the server also logs, in that
SQLite database (in WAL mode, as the feedback's), for each user: the start
of each session and its engine; each input they run, with its result as
text (its output and the first 50 rows of each table, 20,000 characters at
most), whether it failed, how long it took, for a question to the assistant
(``.ask``) the tokens it sent and received, its provider and model and
whose key paid (``page``: the key the user gave in the page; ``server``:
one of the server, of their group or saved in their account), the CPU
seconds and the private
memory of its process after it; and the end of each session, and why
(stopped, restarted, idle, made room for another, ended by an admin, the
server stopped), with the CPU seconds of its process in all and the most
memory of the session, from the samples.  The page tells its users that what they run is logged.  A question to
the assistant (``.ask``) is an input too, logged with its answer; the keys
are never logged.  The console shows the totals
of each user and gives their log as JSON lines (``/admin/activity.jsonl?user=NAME``);
SQLite reads it too::

   sqlite3 activity.sqlite "SELECT created, kind, text, seconds, cpu, memory FROM activity WHERE user = 'bob'"


The activity page
-----------------

With ``--activity-db``, the admins have ``/admin/activity`` (*Activity*,
linked from the console): what the users ran and what it took, read at a
glance.

- **Filters**, one row above everything they scope: the period (24 hours,
  7 days, 30 days, all), a user, an engine.  The view is in the page's
  address, so that it can be kept and shared; times are in the browser's
  time zone.
- **Figures** of the period: the uptime, the users and the sessions they
  started, the inputs and the share that failed, the CPU time, the LLM
  tokens and the share spent on the server's keys, the most memory of one
  session.
- **Uptime**: a strip of the runs of the server over the period (a gap, a
  time it was down; a tick, a run that ended without stopping), and the
  latest runs.  A period starts when the log began, if later: nothing is
  known before, which is not a time the server was down.
- **Over time**: the inputs, the CPU time, the LLM tokens and the active
  users, by hour over 2 days at most, else by day, each in a chart of its
  own on the same time axis, with its table.
- **Users**: each user's inputs, failures, time, CPU, tokens, most memory
  and last input, sortable; choosing one shows only theirs, and a link to
  their whole log.
- **Inputs**: the latest, slowest, or those that took the most CPU or
  tokens, searched by their text, the failed ones only if asked; each
  opens on its result, its memory and the assistant's model and key.
- **Why sessions ended**.

The charts are drawn with `Chart.js <https://www.chartjs.org>`_ (MIT),
which the server serves itself (``tools/serve_repl/vendor/chartjs``): the
page needs nothing from another site.  It follows the computer's light or
dark theme, or the one chosen at its foot.


Raw data and SQL
----------------

With ``--dashboard`` (and ``--activity-db``), the admins also explore the
log at ``/admin/logs/`` (*Raw data and SQL*, linked from the console):
`Datasette
<https://datasette.io/>`_ on the activity log and the ratings
(``--feedback-db``), with its charts (`datasette-vega
<https://github.com/simonw/datasette-vega>`_).  Install them with the
``dashboard`` extra::

   uv run --extra apps --extra tui --extra llm --extra dashboard neurolang-repl-serve \
       --daemon reinstall ... --activity-db activity.sqlite --dashboard

Datasette browses, filters, sorts and searches every table (``activity``,
``runs``, the ratings' ``answers`` and ``feedback``), exports any view as
CSV or JSON, and runs any read-only SQL.  Its queries, on the page of the
activity database, each over the last ``days`` (empty: all):

- *What the users ran, by hour* and *by day*: the inputs, failures, users,
  CPU seconds and the most memory;
- *What each user ran*, *What ran on each engine*;
- *The tokens of the assistant, by user, model and key*, to tell what the
  keys of the server spend;
- *The slowest inputs*, *The inputs that failed* (with their result);
- *Why the sessions ended*;
- *The uptime of the server* (in percent, over 30 days by default) and
  *The runs of the server* (see `Uptime`_).

*Show charting options*, above a result, draws it (a line of the inputs
by hour, a bar of the CPU by user…); the queries by time and by user link
to their charts.

Datasette runs in a process of its own, started and ended with the
server, listening on a unix socket in a directory only the server's user
reads, not on a port; the server passes it the GET requests of the admins
only (the dashboard is read only).


Uptime
------

Whether the server is up, and since when:

- ``GET /health``, open without a login, for a monitor: ``200 {"ok":
  true, "uptime_seconds": N}``, or ``503`` if the sessions can no longer
  start (their builder ended); no answer if the server is down.  Point an
  external monitor at it (`Uptime Kuma <https://github.com/louislam/uptime-kuma>`_,
  UptimeRobot, healthchecks.io…), or a cron job of another machine::

     */5 * * * * curl -fsS --max-time 20 https://SERVER:8443/health > /dev/null || echo "the REPL is down" | mail -s REPL you@example.org

- ``neurolang-repl-serve --daemon status``: what the service manager says,
  whether it runs, since when, how many times it restarted (systemd starts
  it again 5 seconds after it fails) and how its last run ended.
- With ``--activity-db``, each run of the server is logged in its table
  ``runs``: its start, the last minute it was alive and its stop.  A run
  without a stop ended otherwise (killed, crashed, the machine stopped).
  The dashboard's *The uptime of the server* gives the share of the last
  days the runs cover, and *The runs of the server* lists them.  A stop by
  the service manager (``systemctl --user stop``/``restart``) is a stop.


Feedback
--------

With ``--feedback-db feedback.sqlite``, a rating given in the page (*Did
this query do what you wanted?*) is stored on the server, with the query and
its first rows or its error, and, for an answer of the assistant, the
question and its steps, signed by the user's name and groups.  The database
is that of ``neurolang-feedback-server`` (SQLite, in WAL mode, which
sessions write at once): keep it on a local disk, not a network one.  The
console shows the latest ratings and exports, for each engine, the rated
answers of the assistant as YAML, in the form of its examples
(:doc:`query_assistant`): one per question, the query approved by most
ratings.  Without it,
the page sends its ratings where its ``feedback.json`` says, if anywhere.


The service
-----------

``--daemon install`` installs the server as a service of the user who runs
it, started now and at each login (or boot, with linger), serving with the
other options given:

- on Linux, a unit of systemd's user manager,
  :file:`~/.config/systemd/user/neurolang-repl.service`; ``loginctl
  enable-linger $USER`` keeps it running when you are logged out;
- on macOS, a launchd agent,
  :file:`~/Library/LaunchAgents/org.neurolang.repl.plist`.

It runs the Python of the environment that installs it (that of ``uv run``,
the checkout's :file:`.venv/bin/python`; ``--python PATH`` for another),
with that environment's ``PATH``, the paths of the options made absolute,
``--no-browser``, and nothing built again.  ``install`` first checks that
this Python can import what it serves with (it says which extras to add),
checks the certificate, the password file and the keys, and builds the
bundle (without ``--local``), here, where you see the errors; ``reinstall``
does all that before it removes the service installed, which a failure
leaves as it was.  It does not
copy your shell's other variables: an API key in your ``~/.bashrc`` does not
reach it; put it in ``--env-file`` or the users file.

::

   neurolang-repl-serve --daemon install   OPTIONS   # refuses if one is installed
   neurolang-repl-serve --daemon reinstall OPTIONS   # replaces it, with these options,
                                                     # once they are checked and built
   neurolang-repl-serve --daemon remove              # stops and removes it

   neurolang-repl-serve --daemon status              # runs? since when? restarts?
   systemctl --user status neurolang-repl            # Linux
   systemctl --user restart neurolang-repl
   journalctl --user -u neurolang-repl -n 80         # its log
   tail ~/Library/Logs/neurolang-repl.log            # macOS


Security
--------

- Anyone who logs in runs queries and commands with the server's account,
  in its Python: they can read what it can read (the users file included)
  and write files in the data directory, which the sessions share.  Give
  the accounts to people you trust, and run the server as a user of its own.
- Use HTTPS (``--cert``) whenever the server listens on the network: the
  password and the login cookie go over it.
- The server takes a POST only from its own page (in JSON, from its origin),
  so that another site cannot make a logged-in browser change anything.
- There is no limit on the attempts to log in: give long passwords, or put
  the server behind a reverse proxy that limits them.
- Keep the users file, ``--env-file``, the key and the feedback database
  readable by the service's user only.


Behind a reverse proxy
----------------------

The server is the standard library's HTTP server, threaded: enough for a
group, and the queries run in the sessions' processes.  For rate limits,
certificates renewed by the proxy, or a port below 1024, put a proxy in
front, e.g. Caddy::

   repl.example.org {
       reverse_proxy 127.0.0.1:8766 {
           flush_interval -1        # the events of the REPL stream
       }
   }

with the server on ``--host 127.0.0.1 --port 8766`` (no ``--cert``: the
proxy has it) and ``--secure-cookie``, so that the login cookie goes over
HTTPS only.  The proxy must pass the ``Authorization``, ``Cookie``,
``Origin`` and ``Host`` headers unchanged (a POST whose ``Origin`` is not
the ``Host`` is refused), and not buffer ``/api/events`` (a stream of
server-sent events).  Without ``--users`` nor ``--password-file``, give
the proxy's name with ``--server-name repl.example.org``: the server
answers only the names it knows (see `Serving`_).


Troubleshooting
---------------

The service does not start
   ``journalctl --user -u neurolang-repl -n 80``.  ``No module named
   'rich'`` (or another): the environment lacks an extra; ``uv sync --extra
   apps --extra tui --extra llm`` and ``--daemon reinstall``.
The page says *The engine could not start*
   Open *Technical details* on the page; the full traceback is in the
   service's log.  The log tells, before *The REPL is at*, whether each
   engine was prepared; one that was not downloads its data into
   ``--data-dir`` at its first session: check that the server reaches the
   internet (and its sources, OSF and GitHub, are up) and has room.
*The builder of the engines ended*
   The process building the engines died (out of memory, most often): see
   the log, and give the server more memory or ``--light``.
The assistant does not answer
   The console's *AI assistant* column says whose key a user has; *none*
   means that they give theirs in the page.  A key saved on the account
   page or in the console answers at once in the user's open pages, which
   say whose assistant answers under the question's box.
Too many sessions
   Raise ``--max-sessions``, or lower ``--idle-minutes``.
Queries wait for long
   The console's *Sessions* show which are at work: raise ``--max-running``
   if the machine's CPUs are not all busy, or lower ``--query-minutes``.
