import { Container, Nav, Navbar } from 'react-bootstrap';

export default function TopBar({ onChangeTab }: {
  onChangeTab(tab: 'query' | 'dataset'): void,
}) {
  return <Navbar expand='lg' className='bg-primary' data-bs-theme='dark'>
    <Container>
      <Navbar.Brand href=''>Wllama</Navbar.Brand>
      <Navbar.Toggle aria-controls='basic-navbar-nav' />
      <Navbar.Collapse id='basic-navbar-nav'>
        <Nav className='me-auto'>
          {/* <Nav.Link href='https://github.com/ngxson/wllama'>Github</Nav.Link> */}
          <Nav.Link onClick={() => onChangeTab('query')}>Query</Nav.Link>
          <Nav.Link onClick={() => onChangeTab('dataset')}>Dataset</Nav.Link>
        </Nav>
      </Navbar.Collapse>
    </Container>
  </Navbar>;
}