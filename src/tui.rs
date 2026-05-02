use std::future::Future;
use std::io::{self, Stdout, stdout};
use std::path::PathBuf;
use std::pin::pin;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::{Buffer, Color, Line, Modifier, Span, Style, Stylize};
use ratatui::widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph, Widget, Wrap};
use ratatui::{Frame, Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::EmbeddingProgress;

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuiInput {
    pub corpus_path: PathBuf,
    pub query: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Interface {
    corpus_file: String,
    query: String,
    results: Vec<String>,
    status: String,
    completed_batches: usize,
    total_batches: usize,
    submitted: bool,
    input_mode: InputMode,
    running_state: RunningState,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum RunningState {
    #[default]
    Running,
    Done,
}

#[derive(Debug, Default, PartialEq, Eq)]
enum InputMode {
    #[default]
    CorpusFile,
    Query,
    Processing,
    Results,
}

#[derive(Debug, PartialEq, Eq)]
enum Message {
    Input(char),
    Backspace,
    SwitchField,
    Submit,
    Quit,
    Progress(EmbeddingProgress),
}

impl Interface {
    fn new(default_corpus_file: String) -> Self {
        Self {
            corpus_file: default_corpus_file,
            status: "Enter corpus filename, press Tab for query, Enter to run, Esc/q to quit"
                .to_string(),
            ..Self::default()
        }
    }

    fn processing() -> Self {
        Self {
            input_mode: InputMode::Processing,
            status: "Preparing embedding work...".to_string(),
            ..Self::default()
        }
    }

    fn with_results(results: Vec<String>) -> Self {
        Self {
            results,
            input_mode: InputMode::Results,
            status: "Top matches. Press q or Esc to exit.".to_string(),
            ..Self::default()
        }
    }
}

pub async fn collect_input_and_process<T, E, F, Fut>(
    default_corpus_path: PathBuf,
    process: F,
) -> Result<Option<T>, E>
where
    E: From<io::Error>,
    F: FnOnce(TuiInput, UnboundedSender<EmbeddingProgress>) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let _guard = TerminalGuard::enter()?;
    let mut terminal = init_terminal()?;

    let default_corpus_file = default_corpus_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut interface = Interface::new(default_corpus_file);

    while interface.running_state == RunningState::Running {
        terminal.draw(|frame| view(&interface, frame))?;

        if let Some(message) = handle_event()? {
            update(&mut interface, message);
        }
    }

    if !interface.submitted || interface.query.trim().is_empty() {
        return Ok(None);
    }

    let input = TuiInput {
        corpus_path: resolve_corpus_path(default_corpus_path, interface.corpus_file),
        query: interface.query,
    };

    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();
    let mut interface = Interface::processing();
    let future = process(input, progress_tx);
    let mut future = pin!(future);

    loop {
        drain_progress(&mut interface, &mut progress_rx);
        terminal.draw(|frame| view(&interface, frame))?;

        tokio::select! {
            result = &mut future => return result.map(Some),
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                if let Some(Message::Quit) = handle_event_now()? {
                    interface.status = "Processing is still running; quit is disabled during embedding.".to_string();
                }
            }
        }
    }
}

pub fn show_results(results: Vec<String>) -> io::Result<()> {
    let _guard = TerminalGuard::enter()?;
    let mut terminal = init_terminal()?;
    let mut interface = Interface::with_results(results);

    while interface.running_state == RunningState::Running {
        terminal.draw(|frame| view(&interface, frame))?;

        if let Some(message) = handle_event()? {
            update(&mut interface, message);
        }
    }

    Ok(())
}

fn drain_progress(
    interface: &mut Interface,
    progress_rx: &mut UnboundedReceiver<EmbeddingProgress>,
) {
    while let Ok(progress) = progress_rx.try_recv() {
        update(interface, Message::Progress(progress));
    }
}

fn resolve_corpus_path(default_corpus_path: PathBuf, corpus_file: String) -> PathBuf {
    let input_path = PathBuf::from(corpus_file.trim());

    if input_path.is_absolute() {
        return input_path;
    }

    default_corpus_path
        .parent()
        .map(|parent| parent.join(&input_path))
        .unwrap_or(input_path)
}

fn update(interface: &mut Interface, message: Message) {
    match message {
        Message::Input(character) => match interface.input_mode {
            InputMode::CorpusFile => interface.corpus_file.push(character),
            InputMode::Query => interface.query.push(character),
            InputMode::Processing | InputMode::Results => {}
        },
        Message::Backspace => match interface.input_mode {
            InputMode::CorpusFile => {
                interface.corpus_file.pop();
            }
            InputMode::Query => {
                interface.query.pop();
            }
            InputMode::Processing | InputMode::Results => {}
        },
        Message::SwitchField => {
            interface.input_mode = match interface.input_mode {
                InputMode::CorpusFile => InputMode::Query,
                InputMode::Query => InputMode::CorpusFile,
                InputMode::Processing => InputMode::Processing,
                InputMode::Results => InputMode::Results,
            };
        }
        Message::Submit => {
            if matches!(
                interface.input_mode,
                InputMode::Processing | InputMode::Results
            ) || !interface.query.trim().is_empty()
            {
                interface.submitted = true;
                interface.running_state = RunningState::Done;
            } else {
                interface.status = "Query cannot be empty".to_string();
                interface.input_mode = InputMode::Query;
            }
        }
        Message::Quit => {
            if interface.input_mode == InputMode::Processing {
                interface.status =
                    "Processing is still running; quit is disabled during embedding.".to_string();
            } else {
                interface.submitted = false;
                interface.running_state = RunningState::Done;
            }
        }
        Message::Progress(progress) => {
            interface.completed_batches = progress.completed_batches;
            interface.total_batches = progress.total_batches;
            interface.status = progress.message;
        }
    }
}

fn view(interface: &Interface, frame: &mut Frame) {
    let area = frame.area();
    frame.render_widget(interface, area);

    if let Some((x, y)) = interface.cursor_position(area) {
        frame.set_cursor_position((x, y));
    }
}

fn handle_event() -> io::Result<Option<Message>> {
    handle_event_with_timeout(Duration::from_millis(250))
}

fn handle_event_now() -> io::Result<Option<Message>> {
    handle_event_with_timeout(Duration::ZERO)
}

fn handle_event_with_timeout(timeout: Duration) -> io::Result<Option<Message>> {
    if event::poll(timeout)?
        && let Event::Key(key) = event::read()?
        && key.kind == KeyEventKind::Press
    {
        return Ok(handle_key(key));
    }

    Ok(None)
}

fn handle_key(key: KeyEvent) -> Option<Message> {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => Some(Message::Quit),
        KeyCode::Enter => Some(Message::Submit),
        KeyCode::Tab | KeyCode::BackTab => Some(Message::SwitchField),
        KeyCode::Backspace => Some(Message::Backspace),
        KeyCode::Char(character) => Some(Message::Input(character)),
        _ => None,
    }
}

impl Widget for &Interface {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(Line::from(" locursdb ".bold()).centered())
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue));

        Clear.render(area, buf);
        block.render(area, buf);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(5),
                Constraint::Length(2),
            ])
            .split(area);

        match self.input_mode {
            InputMode::Processing => self.render_processing(chunks[2], buf),
            InputMode::Results => self.render_results(chunks[2], buf),
            InputMode::CorpusFile | InputMode::Query => {
                self.render_form(chunks[0], chunks[1], chunks[2], buf)
            }
        }

        Paragraph::new(self.status.as_str().gray())
            .wrap(Wrap { trim: true })
            .render(chunks[3], buf);
    }
}

impl Interface {
    fn cursor_position(&self, area: Rect) -> Option<(u16, u16)> {
        if matches!(self.input_mode, InputMode::Processing | InputMode::Results) {
            return None;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(5),
                Constraint::Length(2),
            ])
            .split(area);

        let (field_area, text) = match self.input_mode {
            InputMode::CorpusFile => (chunks[0], self.corpus_file.as_str()),
            InputMode::Query => (chunks[1], self.query.as_str()),
            InputMode::Processing | InputMode::Results => return None,
        };

        let max_x = field_area.right().saturating_sub(2);
        let cursor_x = field_area
            .x
            .saturating_add(1)
            .saturating_add(text.chars().count() as u16)
            .min(max_x);

        Some((cursor_x, field_area.y.saturating_add(1)))
    }

    fn render_form(&self, corpus_area: Rect, query_area: Rect, help_area: Rect, buf: &mut Buffer) {
        let corpus_style = field_style(self.input_mode == InputMode::CorpusFile);
        let query_style = field_style(self.input_mode == InputMode::Query);

        Paragraph::new(self.corpus_file.as_str())
            .block(
                Block::default()
                    .title(" Corpus file ")
                    .borders(Borders::ALL)
                    .border_style(corpus_style),
            )
            .render(corpus_area, buf);

        Paragraph::new(self.query.as_str())
            .block(
                Block::default()
                    .title(" Query ")
                    .borders(Borders::ALL)
                    .border_style(query_style),
            )
            .render(query_area, buf);

        let help = vec![
            Line::from(vec![
                Span::raw("Tab").blue().bold(),
                Span::raw(" switch field  "),
                Span::raw("Enter").blue().bold(),
                Span::raw(" run search  "),
                Span::raw("Esc/q").blue().bold(),
                Span::raw(" quit"),
            ]),
            Line::from("Relative corpus names are resolved under the configured corpus directory."),
        ];

        Paragraph::new(help)
            .block(Block::default().title(" Help ").borders(Borders::ALL))
            .wrap(Wrap { trim: true })
            .render(help_area, buf);
    }

    fn render_processing(&self, area: Rect, buf: &mut Buffer) {
        let ratio = if self.total_batches == 0 {
            0.0
        } else {
            self.completed_batches as f64 / self.total_batches as f64
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(3),
            ])
            .split(area);

        Gauge::default()
            .block(
                Block::default()
                    .title(" Embedding progress ")
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!(
                "{}/{} batches",
                self.completed_batches, self.total_batches
            ))
            .render(chunks[0], buf);

        let detail = vec![
            Line::from("The interface will stay open while embeddings and search run."),
            Line::from("Detailed traces are also written to log/embedding.log."),
        ];

        Paragraph::new(detail)
            .block(Block::default().title(" Processing ").borders(Borders::ALL))
            .wrap(Wrap { trim: true })
            .render(chunks[1], buf);
    }

    fn render_results(&self, area: Rect, buf: &mut Buffer) {
        let items: Vec<ListItem> = self
            .results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                ListItem::new(vec![
                    Line::from(format!("{}. match", index + 1)).style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Line::from(result.as_str()),
                    Line::from(""),
                ])
            })
            .collect();

        List::new(items)
            .block(Block::default().title(" Results ").borders(Borders::ALL))
            .render(area, buf);
    }
}

fn field_style(active: bool) -> Style {
    if active {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn init_terminal() -> io::Result<Tui> {
    Terminal::new(CrosstermBackend::new(stdout()))
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        execute!(stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = restore_terminal();
    }
}

fn restore_terminal() -> io::Result<()> {
    execute!(stdout(), LeaveAlternateScreen)?;
    disable_raw_mode()
}

pub fn install_panic_hook() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = restore_terminal();
        hook(panic_info);
    }));
}
